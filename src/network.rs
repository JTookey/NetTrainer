use ai_core::{
    layer::{Activation},
    network::{NetworkBuilder, NeuralNetwork},
    AIVec,
};


pub struct TrainingData {
    n_inputs: usize,
    n_outputs: usize,
    length: usize,
    inputs: Vec<f64>,
    outputs: Vec<f64>,
}

impl TrainingData {
    pub fn new() -> Self {
        Self {
            n_inputs: 2,
            n_outputs: 1,
            length: 4,
            inputs: vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            outputs: vec![0.0, 1.0, 1.0, 0.0],
        }
    }

    pub fn get(&self, n: usize, input: &mut AIVec, expectation: &mut AIVec) -> Result<(), &'static str> {
        let index = n % self.length;

        let input_index_from = index * self.n_inputs;
        let input_index_to = input_index_from + self.n_inputs;

        self.inputs[input_index_from..input_index_to].into_iter()
            .zip(input)
            .for_each(|(new, vec)| *vec = *new);

        let output_index_from = index * self.n_outputs;
        let output_index_to = output_index_from + self.n_outputs;

        self.outputs[output_index_from..output_index_to].into_iter()
            .zip(expectation)
           .for_each(|(new, vec)| *vec = *new);

        Ok(())
    }
}

pub struct Network {
    pub nn: NeuralNetwork,
    pub input: AIVec,
    pub output: AIVec,
}

impl Network {
    pub fn new() -> Self {
        let nn = NetworkBuilder::new(2)
            .add_layer(4, Activation::Sigmoid)
            .add_layer(1, Activation::Sigmoid)
            .build()
            .unwrap();

        Self {
            nn,
            input: AIVec::zeros(2), 
            output: AIVec::zeros(1),
        }
    }

    pub fn process(&mut self) -> Result<&AIVec, ai_core::err::AIError> {
        match self.nn.feedforward(&self.input, &mut self.output) {
            Ok(()) => { Ok(&self.output) },
            Err(e) => { 
                Err(e)
         },
        }
    }

    pub fn train(&mut self, expected: &AIVec) -> Result<f64, ai_core::err::AIError> {
        let error = self.nn.calculate_error(&self.input, expected)?;
        self.nn.backproporgate(&self.input, expected)?;

        Ok(error)
    }
}