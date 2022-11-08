from tensorflow_datasets.core import lazy_imports_lib
import tensorflow_datasets as tfds
import numpy as np


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


class AudioFeature(tfds.features.Audio):
  def __init__(self, file_format=None, force_sample_rate=None, force_channels=None, force_samples=None, dtype=None, normalize=False):
    self._normalize = normalize
    channels = 1 if isinstance(force_channels, str) else force_channels
    super().__init__(file_format=file_format, shape=(force_samples, channels), dtype=dtype, sample_rate=force_sample_rate)

  def _encode_file(self, fobj, file_format):
    audio_segment = lazy_imports_lib.lazy_imports.pydub.AudioSegment.from_file(fobj, format=file_format)
    channels = audio_segment.channels

    force_sample_rate = self._sample_rate
    force_samples, force_channels = self._shape

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
        np_dtype = np.dtype(self._dtype.as_numpy_dtype)
        samples = samples_as_dtype(samples, np_dtype)
    if isinstance(force_channels, int) and force_channels > channels:
      # repeat channels until requested number of channels reached
      samples = np.pad(samples, ((0, 0), (0, force_channels - channels)), mode='wrap')
    return samples