"""acoustic_guitar_emotion_recognition dataset."""

import tensorflow_datasets as tfds
import numpy as np
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

_CITATION = """
@Article{acoustic_guitar_emotion_dataset,
  author       = {Turchet, Luca and Pauwels, Johan},
  year         = {2022},
  journaltitle = {IEEE/ACM Transactions on Audio, Speech, and Language Processing},
  title        = {Music Emotion Recognition: Intention of Composers-Performers Versus Perception of Musicians, Non-Musicians, and Listening Machines},
  doi          = {10.1109/taslp.2021.3138709},
  pages        = {305--316},
  volume       = {30},
  issn         = {2329-9304},
  publisher    = {Institute of Electrical and Electronics Engineers ({IEEE})},,
}
"""

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

INSTRUMENT_TYPES = [
  'classical-guitar',
  'steelstring-guitar',
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

PLAYING_TECHNIQUES = [
    'fingers',
    'fingers+harmonics',
    'pick',
    'pick+fingers',
    'pick+hammeron',
    'pick+tapping',
]

MICROPHONE_TYPES = [
  'condenser',
  'condenser+piezo',
  'piezo',
  'piezo+external-condenser'
]

MICROPHONE_POSITIONS = [
  'external',
  'internal',
  'internal+external',
]

class AcousticGuitarEmotionRecognition(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for acoustic_guitar_emotion_recognition dataset."""

  VERSION = tfds.core.Version('0.6.0')
  RELEASE_NOTES = {
      '0.4.0': 'Initial public release.',
      '0.5.0': 'Take file duration into account instead of simply the number of files when grouping performers into splits.',
      '0.6.0': 'Add emotional intensity, microphone type and position as features',
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
            'audio': AudioFeature(force_sample_rate=16000, force_channels='mono', dtype=np.float32, normalize=True),
            'performer': tfds.features.ClassLabel(names=PERFORMERS),
            'instrument_type': tfds.features.ClassLabel(names=INSTRUMENT_TYPES),
            'emotion': tfds.features.ClassLabel(names=EMOTIONS),
            'emotional_intensity': tfds.features.ClassLabel(names=EMOTIONAL_INTENSITIES),
            'playing_technique': tfds.features.ClassLabel(names=PLAYING_TECHNIQUES),
            'microphone_type': tfds.features.ClassLabel(names=MICROPHONE_TYPES),
            'microphone_position': tfds.features.ClassLabel(names=MICROPHONE_POSITIONS),
        }),
        supervised_keys=('audio', 'emotion'),
        homepage='https://www.cimil.disi.unitn.it/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(acoustic_guitar_emotion_recognition): Downloads the data and defines the splits
    # path = dl_manager.download_and_extract('https://todo-data-url')
    zip_path = dl_manager.manual_dir / f'acoustic-guitar-emotion-dataset-v{self.VERSION}.zip'
    if not zip_path.exists():
      raise AssertionError(
        'Cannot find {}, manual download required'.format(zip_path)
      )
    extract_path = dl_manager.extract(zip_path)
    base_dir = extract_path / 'acoustic-guitar'
    with (base_dir / 'annotations_acoustic-guitar.csv').open() as f:
      rows = [row for row in csv.DictReader(f)]

    return {
      'fold1': chain(
        self._generate_examples(base_dir, rows, 1, 37),
        self._generate_examples(base_dir, rows, 73, 109),
        self._generate_examples(base_dir, rows, 212, 224),
      ),
      'fold2': chain(
        self._generate_examples(base_dir, rows, 37, 49),
        self._generate_examples(base_dir, rows, 121, 133),
        self._generate_examples(base_dir, rows, 200, 212),
        self._generate_examples(base_dir, rows, 272, 296),
        self._generate_examples(base_dir, rows, 392, 404),
      ),
      'fold3': chain(
        self._generate_examples(base_dir, rows, 49, 61),
        self._generate_examples(base_dir, rows, 109, 121),
        self._generate_examples(base_dir, rows, 160, 188),
        self._generate_examples(base_dir, rows, 248, 260),
        self._generate_examples(base_dir, rows, 368, 392),
      ),
      'fold4': chain(
        self._generate_examples(base_dir, rows, 61, 73),
        self._generate_examples(base_dir, rows, 145, 160),
        self._generate_examples(base_dir, rows, 188, 200),
        self._generate_examples(base_dir, rows, 236, 248),
        self._generate_examples(base_dir, rows, 332, 356),
      ),
      'fold5': chain(
        self._generate_examples(base_dir, rows, 133, 145),
        self._generate_examples(base_dir, rows, 260, 272),
        self._generate_examples(base_dir, rows, 296, 332),
        self._generate_examples(base_dir, rows, 356, 368),
      ),
    }

  def _generate_examples(self, base_dir, metadata_rows, start_id, end_id):
    """Yields examples."""
    for row in metadata_rows:
      file_id = int(row['file_id'])
      if file_id >= start_id and file_id < end_id:
        full_path = base_dir / row['emotion'] / (row['file_name'] + '.wav')
        example = {'audio': full_path, 'performer': row['performer'], 'instrument_type': row['instrument'], 'emotion': row['emotion'], 'emotional_intensity': row['emotional_intensity'], 'playing_technique': row['playing_technique'], 'microphone_type': row['microphone_type'], 'microphone_position': row['microphone_position']}
        yield file_id, example
