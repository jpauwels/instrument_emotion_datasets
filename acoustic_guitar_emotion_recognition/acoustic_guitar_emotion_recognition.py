"""acoustic_guitar_emotion_recognition dataset."""

import tensorflow_datasets as tfds
import tensorflow.compat.v2 as tf
import sys
from etils import epath
sys.path.append(str(epath.Path(__file__).parent.parent.resolve()))
from audiofeature import AudioFeature
import csv
from itertools import chain

# TODO(acoustic_guitar_emotion_recognition): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(acoustic_guitar_emotion_recognition): BibTeX citation
_CITATION = """
"""

EMOTIONS = [
  'aggressive',
  'relaxed',
  'happy',
  'sad',
]

INSTRUMENT_TYPES = [
  'steelstring_guitar',
  'classical_guitar',
]

PERFORMERS = [
  'LucTur',
  'DavBen',
  'OweWin',
  'ValFui',
  'AdoLaV',
  'MatRig',
  'TomCan',
  'TizCam',
  'SteRom',
  'SimArm',
  'SamLor',
  'AleMar',
  'MasChi',
  'FilMel',
  'GioAcq',
  'TizBol',
  'SalOli',
  'FedCer',
  'CesSam',
  'AntPao',
  'DavRos',
  'FraBen',
  'GiaFer',
  'GioDic',
  'NicCon',
  'AntDel',
  'NicLat',
  'LucFra',
  'AngLoi',
  'MarPia',
]

PLAYING_STYLES = [
    'fingers',
    'fingers_and_harmonics',
    'pick',
    'pick_and_fingers',
    'pick+hammeron',
]


class AcousticGuitarEmotionRecognition(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for acoustic_guitar_emotion_recognition dataset."""

  VERSION = tfds.core.Version('0.4.0')
  RELEASE_NOTES = {
      '0.4.0': 'Initial public release.',
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
            'playing_style': tfds.features.ClassLabel(names=PLAYING_STYLES),
        }),
        supervised_keys=('audio', 'emotion'),
        homepage='https://www.cimil.disi.unitn.it/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(acoustic_guitar_emotion_recognition): Downloads the data and defines the splits
    # path = dl_manager.download_and_extract('https://todo-data-url')
    zip_path = dl_manager.manual_dir / f'acoustic_guitar_emotion_dataset-v{self.VERSION}.zip'
    if not zip_path.exists():
      raise AssertionError(
        'Cannot find {}, manual download required'.format(zip_path)
      )
    extract_path = dl_manager.extract(zip_path)
    base_dir = extract_path / 'emotional_guitar_dataset'
    with tf.io.gfile.GFile(base_dir / 'annotations_emotional_guitar_dataset.csv') as f:
      rows = [row for row in csv.DictReader(f)]

    return {
      'fold1': chain(
        self._generate_examples(base_dir, rows, 1, 13),
        self._generate_examples(base_dir, rows, 37, 49),
        self._generate_examples(base_dir, rows, 160, 176),
        self._generate_examples(base_dir, rows, 200, 212),
        self._generate_examples(base_dir, rows, 260, 272),
        self._generate_examples(base_dir, rows, 284, 296),
      ),
      'fold2': chain(
        self._generate_examples(base_dir, rows, 13, 37),
        self._generate_examples(base_dir, rows, 61, 85),
        self._generate_examples(base_dir, rows, 133, 145),
        self._generate_examples(base_dir, rows, 188, 200),
        self._generate_examples(base_dir, rows, 272, 284),
      ),
      'fold3': chain(
        self._generate_examples(base_dir, rows, 49, 61),
        self._generate_examples(base_dir, rows, 121, 133),
        self._generate_examples(base_dir, rows, 236, 248),
        self._generate_examples(base_dir, rows, 308, 332),
        self._generate_examples(base_dir, rows, 356, 368),
      ),
      'fold4': chain(
        self._generate_examples(base_dir, rows, 85, 97),
        self._generate_examples(base_dir, rows, 109, 121),
        self._generate_examples(base_dir, rows, 145, 160),
        self._generate_examples(base_dir, rows, 176, 188),
        self._generate_examples(base_dir, rows, 332, 356),
      ),
      'fold5': chain(
        self._generate_examples(base_dir, rows, 97, 109),
        self._generate_examples(base_dir, rows, 212, 224),
        self._generate_examples(base_dir, rows, 248, 260),
        self._generate_examples(base_dir, rows, 296, 308),
        self._generate_examples(base_dir, rows, 368, 404),
      ),
    }

  def _generate_examples(self, base_dir, metadata_rows, start_id, end_id):
    """Yields examples."""
    for row in metadata_rows:
      file_id = int(row['file_id'])
      if file_id >= start_id and file_id < end_id:
        example = {'audio': base_dir / row['file_name'], 'emotion': row['emotion'], 'instrument_type': row['instrument'], 'performer': row['composer_pseudonym'], 'playing_style': row['pick/fingers']}
        yield file_id, example
