

use genetic_image_creator::shaper::Shaper;
use image::DynamicImage;

fn main() {
    let source_files_folder_path = "./assets/sources/minecraft";
    //load all source images in
    let mut source_images: Vec<DynamicImage> = vec![];
    for entry in std::fs::read_dir(source_files_folder_path).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        let image = image::open(path).unwrap();
        source_images.push(image);
    }
    let target_image = image::open("./assets/targets/grapes.jpg").unwrap();

    let shaper = Shaper {
        number_of_shapes: 10,
        keep: 10,
        rounds: 3,
        replication_factor: 4,
        source_images,
        target_image,
    };

    let image = shaper.generate_image();

    //save
    image.save("image.png").unwrap();
}
