from config import parse_args
from image_loader import load_image
from app import DrawingApp


def main():
    args = parse_args()
    img = load_image(args.path)

    app = DrawingApp(img, image_path=args.path, output_path=args.output)
    app.run()


if __name__ == '__main__':
    main()
