import matplotlib.pyplot as plt

def disp_frames(frames):
    fig, ax = plt.subplots(1, 4, figsize=(12, 4))
    for col in range(4):
        ax[col].set_title("Frame {}".format(col + 1))
        # Gym env returns image of shape channel x height x width
        # Convert it into channel x width x height for matplotlib
        ax[col].imshow(frames[col].T, cmap='gray')
    fig.tight_layout()
    plt.show()