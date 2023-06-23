"""piano_emotion_recognition dataset."""

import tensorflow_datasets as tfds
import numpy as np
import sys
from etils import epath
sys.path.append(str(epath.Path(__file__).parent.parent.resolve()))
from audiofeature import AudioFeature, MelSpectrogram
import csv
from itertools import chain

# TODO(piano_emotion_recognition): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(piano_emotion_recognition): BibTeX citation
_CITATION = """
"""

PERFORMERS = [
  'BenGul',
  'LucTie',
  'RauMas',
  'MicCal',
  'MicBar',
  'GiaBri',
  'EdoIso',
  'FedSpa',
  'FraPan',
  'TomMag',
  'GiuCar',
  'SavSan',
  'GiaDiT',
  'SimCap',
  'GiaRiz',
  'ManPie',
  'SteDam',
  'LucCen',
  'EnrBis',
  'FraOre',
  'AlbLin',
  'IlaBro',
]

INSTRUMENT_TYPES = [
  'piano',
]

EMOTIONS = [
  'aggressive',
  'relaxed',
  'happy',
  'sad',
]

EMOTIONAL_INTENSITIES = [
  '1',
  '2',
  '3',
]

class PianoEmotionRecognition(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for piano_emotion_recognition dataset."""

  VERSION = tfds.core.Version('0.5.0')
  RELEASE_NOTES = {
      '0.1.0': 'Initial release.',
      '0.2.0': 'Add three more performers.',
      '0.3.0': 'Take file duration into account instead of simply the number of files when grouping performers into splits.',
      '0.4.0': 'Add emotional intensity as feature',
      '0.5.0': 'Add two more performers.',
  }
  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Dowload data manually
  """
  BUILDER_CONFIGS = [
    tfds.core.BuilderConfig(
        name='audio',
        description='Normalized audio representation as float32 in [-1,1], resampled to 16kHz and mixed to mono.',
    ),
    tfds.core.BuilderConfig(
        name='melspectrogram',
        description='Melspectrogram with 96 bands, 512 samples per frame and a step size of 256, computed from a 16kHz, mono waveform.',
    ),
  ]


  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    features = {
      'performer': tfds.features.ClassLabel(names=PERFORMERS),
      'instrument_type': tfds.features.ClassLabel(names=INSTRUMENT_TYPES),
      'emotion': tfds.features.ClassLabel(names=EMOTIONS),
      'emotional_intensity': tfds.features.ClassLabel(names=EMOTIONAL_INTENSITIES),
    }
    if self.builder_config.name == 'audio':
      features['audio'] = AudioFeature(force_sample_rate=16000, force_channels='mono', dtype=np.float32, normalize=True)
    elif self.builder_config.name == 'melspectrogram':
      features['melspectrogram'] = MelSpectrogram(window_size=512, step_size=256, mel_bands=96, force_sample_rate=16000, force_channels='mono', normalize=True)
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict(features),
        supervised_keys=(self.builder_config.name, 'emotion'),
        homepage='https://www.cimil.disi.unitn.it/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(piano_emotion_recognition): Downloads the data and defines the splits
    # path = dl_manager.download_and_extract('https://todo-data-url')
    zip_path = dl_manager.manual_dir / f'piano-emotion-dataset-v{self.VERSION}.zip'
    if not zip_path.exists():
      raise AssertionError(
        'Cannot find {}, manual download required'.format(zip_path)
      )
    extract_path = dl_manager.extract(zip_path)
    base_dir = extract_path / 'piano'
    with (base_dir / 'annotations_piano.csv').open() as f:
      rows = [row for row in csv.DictReader(f)]

    return {
      'fold1': chain(
        self._generate_examples(base_dir, rows, 1, 13),
        self._generate_examples(base_dir, rows, 56, 68),
        self._generate_examples(base_dir, rows, 203, 227),
        self._generate_examples(base_dir, rows, 287, 299),
      ),
      'fold2': chain(
        self._generate_examples(base_dir, rows, 13, 31),
        self._generate_examples(base_dir, rows, 43, 56),
        self._generate_examples(base_dir, rows, 227, 251),
      ),
      'fold3': chain(
        self._generate_examples(base_dir, rows, 31, 43),
        self._generate_examples(base_dir, rows, 86, 100),
        self._generate_examples(base_dir, rows, 131, 155),
        self._generate_examples(base_dir, rows, 191, 203),
      ),
      'fold4': chain(
        self._generate_examples(base_dir, rows, 68, 86),
        self._generate_examples(base_dir, rows, 155, 167),
        self._generate_examples(base_dir, rows, 179, 191),
        self._generate_examples(base_dir, rows, 263, 275),
      ),
      'fold5': chain(
        self._generate_examples(base_dir, rows, 100, 131),
        self._generate_examples(base_dir, rows, 167, 179),
        self._generate_examples(base_dir, rows, 251, 263),
        self._generate_examples(base_dir, rows, 275, 287),
      ),
    }

  def _generate_examples(self, base_dir, metadata_rows, start_id, end_id):
    """Yields examples."""
    for row in metadata_rows:
      file_id = int(row['file_id'])
      if file_id >= start_id and file_id < end_id:
        full_path = base_dir / row['emotion'] / (row['file_name'] + '.wav')
        example = {self.builder_config.name: full_path, 'performer': row['performer'], 'instrument_type': row['instrument'], 'emotion': row['emotion'], 'emotional_intensity': row['emotional_intensity']}
        yield file_id, example
