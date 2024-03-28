# Copyright The Orthority Contributors.
#
# This file is part of Orthority.
#
# Orthority is free software: you can redistribute it and/or modify it under the terms of the GNU
# Affero General Public License as published by the Free Software Foundation, either version 3 of
# the License, or (at your option) any later version.
#
# Orthority is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License along with Orthority.
# If not, see <https://www.gnu.org/licenses/>.

"""Factories for creating camera objects from parameter files."""
from __future__ import annotations

from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
from typing import IO, Sequence

import rasterio as rio

from orthority import param_io
from orthority.camera import Camera, create_camera, FrameCamera
from orthority.errors import CrsMissingError, ParamError
from orthority.utils import get_filename, join_ofile, OpenFile


class Cameras(ABC):
    """Base camera factory class."""

    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @property
    @abstractmethod
    def filenames(self) -> set[str]:
        """Filenames for which there are cameras."""
        pass

    @abstractmethod
    def get(self, filename: str | PathLike | OpenFile) -> Camera:
        """
        Create and return the camera object for the given image filename.

        :param filename:
            Image filename.  Can be a path, URI string, or :class:`~fsspec.core.OpenFile`
            instance.

        :return:
            Camera object.
        """
        pass

    @abstractmethod
    def write_param(self, out_dir: str | PathLike | OpenFile, overwrite: bool = False):
        """
        Write camera parameters to Orthority format file(s).

        :param out_dir:
            Directory to write into.  Can be a path, URI string, or an
            :class:`~fsspec.core.OpenFile` object.
        :param overwrite:
            Whether to overwrite file(s) if they exist.
        """
        pass


class FrameCameras(Cameras):
    """
    Frame camera factory for interior and exterior parameter files.

    :param int_param:
        Interior parameter file or dictionary.  If a file, can be a path or URI string,
        an :class:`~fsspec.core.OpenFile` object or a file object, opened in text mode ('rt').
    :param ext_param:
        Exterior parameter file or dictionary.  If a file, can be a path or URI string,
        an :class:`~fsspec.core.OpenFile` object or a file object, opened in text mode ('rt').
    :param io_kwargs:
        Optional dictionary of keyword arguments for the :class:`~orthority.param_io.Reader`
        sub-class corresponding to the exterior (and possibly interior) parameter file format.
        If ``ext_param`` is a dictionary, these arguments are not passed to a
        :class:`~orthority.param_io.Reader`, but a 'crs' argument is used for
        :attr:`FrameCameras.crs`.
    :param cam_kwargs:
        Optional dictionary of keyword arguments for the :class:`~orthority.camera.FrameCamera`
        sub-class corresponding to the interior parameters.  Should exclude the interior parameters
        themselves which are read from ``int_param``.
    """

    # TODO: add doc link to file and dictionary formats.
    def __init__(
        self,
        int_param: str | PathLike | OpenFile | IO[str] | dict[str, dict],
        ext_param: str | PathLike | OpenFile | IO[str] | dict[str, dict],
        io_kwargs: dict = None,
        cam_kwargs: dict = None,
    ):
        self._int_param_dict, self._ext_param_dict, self._crs = self._read_param(
            int_param, ext_param, **(io_kwargs or {})
        )
        self._cam_kwargs = cam_kwargs or {}
        self._cameras = {}

    @property
    def crs(self) -> rio.CRS | None:
        """CRS of the world coordinate system."""
        return self._crs

    @property
    def filenames(self) -> set[str]:
        return set(self._ext_param_dict.keys())

    @staticmethod
    def _read_param(
        int_param: str | PathLike | OpenFile | IO[str] | dict[str, dict],
        ext_param: str | PathLike | OpenFile | IO[str] | dict[str, dict],
        **kwargs,
    ) -> tuple[dict[str, dict], dict[str, dict], rio.CRS | None]:
        """Read interior parameters, exterior parameters and world CRS."""
        crs = None
        if not isinstance(int_param, dict):
            # read interior params
            int_param_suffix = Path(get_filename(int_param)).suffix.lower()
            int_param_dict = None
            if int_param_suffix in ['.yaml', '.yml']:
                int_param_dict = param_io.read_oty_int_param(int_param)
            elif int_param_suffix == '.json':
                # only read OSfM interior params here if the interior and exterior param files
                # are not the same, otherwise they are read with exteriors below
                if int_param != ext_param:
                    # TODO: does != work for all file object types
                    int_param_dict = param_io.read_osfm_int_param(int_param)
            else:
                raise ParamError(f"'{int_param_suffix}' file type not supported.")
        else:
            int_param_dict = int_param

        if not isinstance(ext_param, dict):
            # read exterior params and CRS
            ext_param_suffix = Path(get_filename(ext_param)).suffix.lower()
            if ext_param_suffix in ['.csv', '.txt']:
                reader = param_io.CsvReader(ext_param, **kwargs)
            elif ext_param_suffix == '.json':
                reader = param_io.OsfmReader(ext_param, **kwargs)
            elif ext_param_suffix == '.geojson':
                reader = param_io.OtyReader(ext_param)
            else:
                raise ParamError(f"'{ext_param_suffix}' file type not supported.")
            ext_param_dict = reader.read_ext_param()
            crs = reader.crs

            # read interior params if not read already
            int_param_dict = int_param_dict or reader.read_int_param()
        else:
            ext_param_dict = ext_param

        # copy crs from kwargs if it is not set already
        crs = crs or kwargs.get('crs', None)
        crs = rio.CRS.from_string(crs) if isinstance(crs, str) else crs

        return int_param_dict, ext_param_dict, crs

    def get(self, filename: str | PathLike | OpenFile) -> FrameCamera:
        # get exterior params for filename
        filename = Path(get_filename(filename))
        ext_param = self._ext_param_dict.get(
            filename.name, self._ext_param_dict.get(filename.stem, None)
        )
        if not ext_param:
            raise ParamError(f"Could not find exterior parameters for '{filename.name}'.")

        # get interior params for ext_param
        cam_id = ext_param.get('camera', None)
        if cam_id:
            if cam_id not in self._int_param_dict:
                raise ParamError(f"Could not find interior parameters for camera '{cam_id}'.")
            int_param = self._int_param_dict[cam_id]
        elif len(self._int_param_dict) == 1:
            int_param = list(self._int_param_dict.values())[0]
        else:
            raise ParamError(
                f"Exterior parameters for '{filename.name}' should define a 'camera' ID."
            )

        # create camera on first use
        if cam_id not in self._cameras:
            self._cameras[cam_id] = create_camera(**int_param, **self._cam_kwargs)

        # update exterior params
        self._cameras[cam_id].update(xyz=ext_param['xyz'], opk=ext_param['opk'])

        return self._cameras[cam_id]

    def write_param(self, out_dir: str | PathLike | OpenFile, overwrite: bool = False):
        # write interior params
        int_param_file = join_ofile(out_dir, 'int_param.yaml', mode='wt')
        param_io.write_int_param(int_param_file, self._int_param_dict, overwrite=overwrite)

        if not self.crs:
            raise CrsMissingError("A world 'crs' is required to write exterior parameters.")

        # write exterior params
        ext_param_file = join_ofile(out_dir, 'ext_param.geojson', mode='wt')
        param_io.write_ext_param(
            ext_param_file, self._ext_param_dict, overwrite=overwrite, crs=self.crs
        )


class ExifCameras(FrameCameras):
    """
    Frame camera factory for image file(s) with EXIF / XMP tags.

    :param files:
        Image file(s) to read as a tuple of paths or URI strings, :class:`~fsspec.core.OpenFile`
        objects in binary mode ('rb'), or dataset readers.
    :param io_kwargs:
        Optional dictionary of keyword arguments for the :class:`~orthority.param_io.Reader`
        sub-class corresponding to the exterior (and possibly interior) parameter file format.
        If ``ext_param`` is a dictionary, these arguments are not passed to a
        :class:`~orthority.param_io.Reader`, but a 'crs' argument is used for
        :attr:`FrameCameras.crs`.
    :param cam_kwargs:
        Optional dictionary of keyword arguments for the :class:`~orthority.camera.FrameCamera`
        sub-class corresponding to the interior parameters.  Should exclude the interior parameters
        themselves which are read from ``int_param``.
    """

    def __init__(
        self,
        files: Sequence[str | PathLike | OpenFile | rio.DatasetReader],
        io_kwargs: dict = None,
        cam_kwargs: dict = None,
    ):
        # TODO: read exif tags and create cameras as needed rather than upfront (would
        #  complicate exporting which needs upfront params)?
        io_kwargs = io_kwargs or {}
        reader = param_io.ExifReader(files, **io_kwargs)
        int_param_dict = reader.read_int_param()
        ext_param_dict = reader.read_ext_param()
        io_kwargs.update(crs=reader.crs)
        super().__init__(int_param_dict, ext_param_dict, io_kwargs=io_kwargs, cam_kwargs=cam_kwargs)
