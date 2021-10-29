import numpy as np

def gen_fake_data():
    fake_data = np.random.rand(1, 3, 480, 480).astype(np.float32)
    np.save("fake_data.npy", fake_data)


def gen_fake_label():
    fake_label = np.random.randint(-1, 18, [1, 480, 480]).astype(np.float32)
    print(fake_label)
    np.save("fake_label.npy", fake_label)

if __name__ == "__main__":
    gen_fake_data()
    gen_fake_label()