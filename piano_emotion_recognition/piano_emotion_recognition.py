"""piano_emotion_recognition dataset."""

import tensorflow_datasets as tfds
import tensorflow.compat.v2 as tf
import sys
from etils import epath
sys.path.append(str(epath.Path(__file__).parent.parent.resolve()))
from audiofeature import AudioFeature
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

EMOTIONS = [
  'aggressive',
  'relaxed',
  'happy',
  'sad',
]

INSTRUMENT_TYPES = [
  'piano',
]

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
]


class PianoEmotionRecognition(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for piano_emotion_recognition dataset."""

  VERSION = tfds.core.Version('0.0.1')
  RELEASE_NOTES = {
      '0.0.1': 'Initial release.',
  }
  MANUAL_DOWNLOAD_INSTRUCTIONS = """
  Dowload data manually
  """

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'audio': AudioFeature(force_sample_rate=16000, force_channels='mono', dtype=tf.float32, normalize=True),
            'emotion': tfds.features.ClassLabel(names=EMOTIONS),
            'instrument_type': tfds.features.ClassLabel(names=INSTRUMENT_TYPES),
            'performer': tfds.features.ClassLabel(names=PERFORMERS),
        }),
        supervised_keys=('audio', 'emotion'),
        homepage='https://www.cimil.disi.unitn.it/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(piano_emotion_recognition): Downloads the data and defines the splits
    # path = dl_manager.download_and_extract('https://todo-data-url')
    zip_path = dl_manager.manual_dir / f'piano_emotion_dataset-v{self.VERSION}.zip'
    if not zip_path.exists():
      raise AssertionError(
        'Cannot find {}, manual download required'.format(zip_path)
      )
    extract_path = dl_manager.extract(zip_path)
    base_dir = extract_path / 'piano'
    with tf.io.gfile.GFile(base_dir / 'annotations_piano.csv') as f:
      rows = [row for row in csv.DictReader(f)]

    return {
      'fold1': chain(
        self._generate_examples(base_dir, rows, 1, 31),
        self._generate_examples(base_dir, rows, 191, 203),
        self._generate_examples(base_dir, rows, 227, 239),
      ),
      'fold2': chain(
        self._generate_examples(base_dir, rows, 31, 43),
        self._generate_examples(base_dir, rows, 100, 119),
        self._generate_examples(base_dir, rows, 167, 179),
      ),
      'fold3': chain(
        self._generate_examples(base_dir, rows, 43, 56),
        self._generate_examples(base_dir, rows, 86, 100),
        self._generate_examples(base_dir, rows, 143, 155),
        self._generate_examples(base_dir, rows, 179, 191),
      ),
      'fold4': chain(
        self._generate_examples(base_dir, rows, 56, 68),
        self._generate_examples(base_dir, rows, 131, 143),
        self._generate_examples(base_dir, rows, 155, 167),
        self._generate_examples(base_dir, rows, 203, 215),
      ),
      'fold5': chain(
        self._generate_examples(base_dir, rows, 68, 86),
        self._generate_examples(base_dir, rows, 119, 131),
        self._generate_examples(base_dir, rows, 215, 227),
      ),
    }

  def _generate_examples(self, base_dir, metadata_rows, start_id, end_id):
    """Yields examples."""
    for row in metadata_rows:
      file_id = int(row['file_name'].split('_')[0])
      if file_id >= start_id and file_id < end_id:
        full_path = base_dir / row['emotion'] / row['file_name']
        example = {'audio': full_path, 'emotion': row['emotion'], 'instrument_type': row['instrument'], 'performer': row['musician_pseudonym']}
        yield file_id, example
