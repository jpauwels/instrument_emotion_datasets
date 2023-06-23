"""electric_guitar_emotion_recognition dataset."""

import tensorflow_datasets as tfds
import numpy as np
import sys
from etils import epath
sys.path.append(str(epath.Path(__file__).parent.parent.resolve()))
from audiofeature import AudioFeature
import csv
from itertools import chain

# TODO(electric_guitar_emotion_recognition): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(electric_guitar_emotion_recognition): BibTeX citation
_CITATION = """
"""

PERFORMERS = [
  'MatPoz',
  'GioSca',
  'ThoBor',
  'TizBol',
  'PhiRom',
  'PaoTad',
  'AdoLav',
  'NisSil',
  'AleMar',
  'RobBia',
  'AntPao',
  'DavBen',
  'MicRos',
  'NicSal',
  'NicCon',
  'ValFui',
  'DavAlt',
  'DavPor',
  'MarFio',
  'TomCost',
  'SalGiu',
  'SimLuc',
  'GiuMel',
  'DieKod',
  'LucSca',
  'LucCol',
  'MarTed',
  'RicSer',
]

INSTRUMENT_TYPES = [
  'electric-guitar',
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


class ElectricGuitarEmotionRecognition(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for electric_guitar_emotion_recognition dataset."""

  VERSION = tfds.core.Version('0.5.0')
  RELEASE_NOTES = {
      '0.1.0': 'Initial release.',
      '0.2.0': 'Add nine more performers.',
      '0.3.0': 'Take file duration into account instead of simply the number of files when grouping performers into splits.',
      '0.4.0': 'Add emotional intensity as feature',
      '0.5.0': 'Add one more performer.',
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
        }),
        supervised_keys=('audio', 'emotion'),
        homepage='https://www.cimil.disi.unitn.it/',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(electric_guitar_emotion_recognition): Downloads the data and defines the splits
    # path = dl_manager.download_and_extract('https://todo-data-url')
    zip_path = dl_manager.manual_dir / f'electric-guitar-emotion-dataset-v{self.VERSION}.zip'
    if not zip_path.exists():
      raise AssertionError(
        'Cannot find {}, manual download required'.format(zip_path)
      )
    extract_path = dl_manager.extract(zip_path)
    base_dir = extract_path / 'electric-guitar'
    with (base_dir / 'annotations_electric-guitar.csv').open() as f:
      rows = [row for row in csv.DictReader(f)]

    return {
      'fold1': chain(
        self._generate_examples(base_dir, rows, 1, 13),
        self._generate_examples(base_dir, rows, 25, 37),
        self._generate_examples(base_dir, rows, 173, 186),
        self._generate_examples(base_dir, rows, 210, 222),
        self._generate_examples(base_dir, rows, 270, 282),
        self._generate_examples(base_dir, rows, 354, 366),
      ),
      'fold2': chain(
        self._generate_examples(base_dir, rows, 13, 25),
        self._generate_examples(base_dir, rows, 73, 97),
        self._generate_examples(base_dir, rows, 137, 149),
        self._generate_examples(base_dir, rows, 186, 198),
        self._generate_examples(base_dir, rows, 246, 258),
      ),
      'fold3': chain(
        self._generate_examples(base_dir, rows, 37, 49),
        self._generate_examples(base_dir, rows, 97, 109),
        self._generate_examples(base_dir, rows, 133, 137),
        self._generate_examples(base_dir, rows, 161, 173),
        self._generate_examples(base_dir, rows, 198, 210),
        self._generate_examples(base_dir, rows, 234, 246),
        self._generate_examples(base_dir, rows, 282, 294),
      ),
      'fold4': chain(
        self._generate_examples(base_dir, rows, 49, 73),
        self._generate_examples(base_dir, rows, 109, 133),
        self._generate_examples(base_dir, rows, 222, 234),
        self._generate_examples(base_dir, rows, 306, 318),
        self._generate_examples(base_dir, rows, 330, 342),
      ),
      'fold5': chain(
        self._generate_examples(base_dir, rows, 149, 161),
        self._generate_examples(base_dir, rows, 258, 270),
        self._generate_examples(base_dir, rows, 294, 306),
        self._generate_examples(base_dir, rows, 318, 330),
        self._generate_examples(base_dir, rows, 342, 354),
        self._generate_examples(base_dir, rows, 366, 378),
      ),
    }

  def _generate_examples(self, base_dir, metadata_rows, start_id, end_id):
    """Yields examples."""
    for row in metadata_rows:
      file_id = int(row['file_id'])
      if file_id >= start_id and file_id < end_id:
        full_path = base_dir / row['emotion'] / (row['file_name'] + '.wav')
        example = {'audio': full_path, 'performer': row['performer'], 'instrument_type': row['instrument'], 'emotion': row['emotion'], 'emotional_intensity': row['emotional_intensity']}
        yield file_id, example
