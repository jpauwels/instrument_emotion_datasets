"""electric_guitar_emotion_recognition dataset."""

import tensorflow_datasets as tfds
import tensorflow.compat.v2 as tf
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

EMOTIONS = [
  'aggressive',
  'relaxed',
  'happy',
  'sad',
]

INSTRUMENT_TYPES = [
  'electric_guitar',
]

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
]


class ElectricGuitarEmotionRecognition(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for electric_guitar_emotion_recognition dataset."""

  VERSION = tfds.core.Version('0.0.2')
  RELEASE_NOTES = {
      '0.0.1': 'Initial release.',
      '0.0.2': 'Add nine more performers.',
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
    # TODO(electric_guitar_emotion_recognition): Downloads the data and defines the splits
    # path = dl_manager.download_and_extract('https://todo-data-url')
    zip_path = dl_manager.manual_dir / f'electric_guitar_emotion_dataset-v{self.VERSION}.zip'
    if not zip_path.exists():
      raise AssertionError(
        'Cannot find {}, manual download required'.format(zip_path)
      )
    extract_path = dl_manager.extract(zip_path)
    base_dir = extract_path / 'electric_guitar'
    with tf.io.gfile.GFile(base_dir / 'annotations_electric_guitar.csv') as f:
      rows = [row for row in csv.DictReader(f)]

    return {
      'fold1': chain(
        self._generate_examples(base_dir, rows, 1, 13),
        self._generate_examples(base_dir, rows, 61, 85),
        self._generate_examples(base_dir, rows, 97, 109),
        self._generate_examples(base_dir, rows, 173, 186),
        self._generate_examples(base_dir, rows, 234, 246),
      ),
      'fold2': chain(
        self._generate_examples(base_dir, rows, 13, 25),
        self._generate_examples(base_dir, rows, 85, 97),
        self._generate_examples(base_dir, rows, 137, 149),
        self._generate_examples(base_dir, rows, 186, 198),
        self._generate_examples(base_dir, rows, 294, 306),
        self._generate_examples(base_dir, rows, 354, 366),
      ),
      'fold3': chain(
        self._generate_examples(base_dir, rows, 25, 37),
        self._generate_examples(base_dir, rows, 161, 173),
        self._generate_examples(base_dir, rows, 222, 234),
        self._generate_examples(base_dir, rows, 258, 270),
        self._generate_examples(base_dir, rows, 282, 294),
        self._generate_examples(base_dir, rows, 306, 318),
      ),
      'fold4': chain(
        self._generate_examples(base_dir, rows, 37, 49),
        self._generate_examples(base_dir, rows, 149, 161),
        self._generate_examples(base_dir, rows, 198, 222),
        self._generate_examples(base_dir, rows, 330, 354),
      ),
      'fold5': chain(
        self._generate_examples(base_dir, rows, 49, 61),
        self._generate_examples(base_dir, rows, 109, 137),
        self._generate_examples(base_dir, rows, 246, 258),
        self._generate_examples(base_dir, rows, 270, 282),
        self._generate_examples(base_dir, rows, 318, 330),
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
