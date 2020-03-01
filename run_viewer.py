import hydra

from frcnn.viewer import ImageViewer


@hydra.main(config_path="./config/viewer.yml")
def main(cfg):
    image_viewer = ImageViewer(cfg)
    image_viewer.run()


if __name__ == "__main__":
    main()
