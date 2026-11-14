

import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
path = str(ROOT / "data" / "experiment1" / "train.csv"),
df = pd.read_csv(path)
print(df.head())