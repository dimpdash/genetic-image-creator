use ferris_says::say;

use image::{DynamicImage, GenericImage, GenericImageView};
use imageproc;

fn rectangle_image(width: u32, height: u32) -> DynamicImage {
    let mut image = DynamicImage::new_rgba8(width, height);
    for x in 0..width {
        for y in 0..height {
            image.put_pixel(x, y, image::Rgba([255, 0, 0, 255]));
        }
    }
    image
}

//Shape struct
#[derive(Clone)]
struct Shape {
    x: i64,
    y: i64,
    rot: f64,
    image: DynamicImage,
    fit: f64,
}

impl Shape {
    fn mutate(&mut self) {
        self.x = self.x + 1;
        self.y = self.y + 1;
    }
}

struct Shaper {
    number_of_shapes: u32,
    keep: usize,
    rounds: u32,
    replication_factor: u32,
}

impl Shaper {
    fn get_random_shape(&self) -> Shape {
        let shape = Shape {
            x: 0,
            y: 0,
            rot: 0.0,
            image: rectangle_image(10, 10),
            fit: 0.0,
        };
        shape
    }

    fn add_best_shape(&self, target_image: &DynamicImage, image: &mut DynamicImage) {
        let mut shapes: Vec<Shape> = vec![];

        // generate initial shapes
        for _ in 0..self.number_of_shapes {
            let shape = self.get_random_shape();
            shapes.push(shape);
        }

        // rounds
        for i in 0..self.rounds {
            println!("Round {}", i);
            // mutate shapes
            for shape in shapes.iter_mut() {
                let mut image_copy = image.clone();
                //add shape to image
                image::imageops::overlay(&mut image_copy, &shape.image, shape.x, shape.y);

                println!("Calculating fit");
                //compare to target image
                let mse = imageproc::stats::root_mean_squared_error(target_image, &image_copy);
                shape.fit = mse;
            }

            //sort shapes based on fit with with lowest first
            shapes.sort_by(|a, b| a.fit.total_cmp(&b.fit));

            // only keep some shapes
            shapes.truncate(self.keep);

            let mut new_shapes = vec![];

            // mutate shapes
            for shape in shapes.iter_mut() {
                for _ in 0..self.replication_factor {
                    // mutate shape
                    let mut copy_shape = shape.clone();
                    copy_shape.mutate();
                    new_shapes.push(copy_shape);
                }
            }

            shapes.append(&mut new_shapes);
        }

        //overlay the best shape onto the image
        let best_shape = shapes.first().unwrap();
        image::imageops::overlay(image, &best_shape.image, best_shape.x, best_shape.y);
    }

    fn generate_image(&self, target_image: &DynamicImage) -> DynamicImage {
        // create a new empty image of the same size
        let mut image = DynamicImage::new_rgba8(target_image.width(), target_image.height());

        //add shape
        for _ in 0..self.number_of_shapes {
            println!("Adding shape");
            self.add_best_shape(target_image, &mut image);
        }

        image
    }
}

fn main() {
    //open target image
    let target_image = image::open("./assets/targets/grapes.jpg").unwrap();

    let shaper = Shaper {
        number_of_shapes: 3,
        keep: 2,
        rounds: 10,
        replication_factor: 1,
    };
    let image = shaper.generate_image(&target_image);

    //save
    image.save("image.png").unwrap();
}
