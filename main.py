from src.data.loader import CMAPSSLoader
from configs.config import TRAIN_FILE, TEST_FILE, RUL_FILE

loader = CMAPSSLoader(
    train_path=TRAIN_FILE,
    test_path=TEST_FILE,
    rul_path=RUL_FILE
)

train = loader.load_train()
test = loader.load_test()
rul = loader.load_rul()

print(train.shape)
print(test.shape)
print(rul.shape)