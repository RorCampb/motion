mod world;
use world::{Body, World};

fn main() {
    let mut world = World {
        bodies: vec![
            Body {
                id: 1,
                position: vec![0.0, 1.3, 4.0],
                velocity: vec![2.3, 4.0, 0.1],
            },
            Body {
                id: 2,
                position: vec![0.5, 1.7, 4.0],
                velocity: vec![1.3, 2.0, 0.1],
            }, 
        ],
        dim: 3,
        time: 0.0
    };

    for _ in 0..10 {
        world.step(0.1);
        println!("{:#?}", world);
    }
}
 
