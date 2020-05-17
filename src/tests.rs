use crate::NN;

#[test]
fn are_layers_correct() {
    let layers: Vec<usize> = vec![2, 3, 1];
    let nn = NN { layers: &layers, bias: None, weights: None };
    assert_eq!(nn.layers.len(), 3);
}

#[test]
fn are_weights_right_shape() {
    let layers: Vec<usize> = vec![2, 3, 1];
    let mut nn = NN { layers: &layers, bias: None, weights: None };
    nn.init();
    match nn.weights {
        Some(w) => {
            assert_eq!(w.len(), layers[1..].len());
            assert_eq!(w[0].len(), layers[0]);
            assert_eq!(w[0][0].len(), layers[1]);
            assert_eq!(w[1][0].len(), layers[2]);
        },
        None => panic!("Something's wrong I can feel it.")
    }
}
