from typing import BinaryIO, Optional, Union

from tensorflow_datasets.core import lazy_imports_lib
from tensorflow_datasets.core import utils
from tensorflow_datasets.core.features import feature as feature_lib
from tensorflow_datasets.core.features import tensor_feature
from tensorflow_datasets.core.features.audio_feature import Audio
from tensorflow_datasets.core.utils import type_utils
import numpy as np

Encoding = tensor_feature.Encoding


def samples_as_dtype(np_array, np_dtype):
  if np.issubdtype(np_array.dtype, np.integer) and np.issubdtype(np_dtype, np.floating):
    # Convert int to float
    bitdepth = 8 * np_array.dtype.itemsize
    peak_value = 1 << (bitdepth-1)
    if np.issubdtype(np_array.dtype, np.unsignedinteger):
      return ((np_array - peak_value + 1) / peak_value).astype(np_dtype)
    else:
      return (np_array / peak_value).astype(np_dtype)
  elif np.issubdtype(np_array.dtype, np.floating) and np.issubdtype(np_dtype, np.integer):
    # Convert float to int
    bitdepth = 8 * np_dtype.itemsize
    peak_value = 1 << (bitdepth-1)
    if np.issubdtype(np_dtype, np.unsignedinteger):
      return ((np_array + 1) * peak_value - 1).astype(np_dtype)
    else:
      return (np_array * peak_value).astype(np_dtype)
  else:
    # Convert int to int or float to float
    return np_array.astype(np_dtype)


def mix_to_mono(audio_segments):
  def mix_next_segment(downmixed_segment, segments_to_add):
    if segments_to_add:
      return mix_next_segment(downmixed_segment.overlay(segments_to_add[0]), segments_to_add[1:])
    else:
      return downmixed_segment
  gain = lazy_imports_lib.lazy_imports.pydub.utils.ratio_to_db(1 / len(audio_segments))
  segments = [s.apply_gain(gain) for s in audio_segments]
  return mix_next_segment(segments[0], segments[1:])


class AudioFeature(Audio):
  def __init__(
    self,
    *,
    file_format: Optional[str] = None,
    shape: utils.Shape = (None,),
    dtype: type_utils.TfdsDType = np.int64,
    force_sample_rate: Optional[int] = None,
    encoding: Union[str, Encoding] = Encoding.NONE,
    doc: feature_lib.DocArg = None,
    lazy_decode: bool = False,
    force_channels: Optional[int] = None,
    normalize: bool = False,
  ):
    self._normalize = normalize
    if isinstance(force_channels, str) or force_channels is None or force_channels == 1:
      shape = (shape[0],)
    else:
      shape = (shape[0], force_channels)
    super().__init__(
      file_format=file_format,
      shape=shape,
      dtype=dtype,
      sample_rate=force_sample_rate,
      encoding=encoding,
      doc=doc,
      lazy_decode=lazy_decode,
    )
    if not lazy_decode:
      self._audio_decoder.encode_audio = self._eager_encode_audio

  def _eager_encode_audio(
    self, fobj: BinaryIO, file_format: Optional[str]
  ) -> np.ndarray:
    audio_segment = lazy_imports_lib.lazy_imports.pydub.AudioSegment.from_file(fobj, format=file_format)
    channels = audio_segment.channels

    force_sample_rate = self._sample_rate
    force_samples = self._shape[0]
    force_channels = self._audio_decoder._channels

    if force_channels == 'first':
      audio_segment = audio_segment.split_to_mono()[0]
    elif force_channels == 'last':
      audio_segment = audio_segment.split_to_mono()[-1]
    elif isinstance(force_channels, int) and force_channels < channels:
      mono_segments = audio_segment.split_to_mono()[:force_channels]
      audio_segment = lazy_imports_lib.lazy_imports.pydub.AudioSegment.from_mono_audiosegments(*mono_segments)
    if force_sample_rate is not None and audio_segment.frame_rate != force_sample_rate:
      audio_segment = audio_segment.set_frame_rate(force_sample_rate)
    if force_samples:
      force_duration_ms = 1000 * force_samples / audio_segment.frame_rate
      duration_ms = len(audio_segment)
      if force_duration_ms <= duration_ms:
        audio_segment = audio_segment[:force_duration_ms]
      else:
        silence = lazy_imports_lib.lazy_imports.pydub.AudioSegment.silent(
          duration=duration_ms-force_duration_ms, frame_rate=audio_segment.frame_rate
        )
        audio_segment += silence
    if force_channels == 'mono':
      audio_segment = mix_to_mono(audio_segment.split_to_mono())
    if self._normalize:
      audio_segment = audio_segment.remove_dc_offset().normalize()
    samples = np.array([s.get_array_of_samples() for s in audio_segment.split_to_mono()]).T
    if self._dtype:
      samples = samples_as_dtype(samples, self._dtype)
    if isinstance(force_channels, int) and force_channels > channels:
      # repeat channels until requested number of channels reached
      samples = np.pad(samples, ((0, 0), (0, force_channels - channels)), mode='wrap')
    return np.squeeze(samples)
