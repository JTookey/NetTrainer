use std::fmt::Write as FmtWrite;

use ai_core::{
    layer::{Activation},
    network::{NetworkBuilder, NeuralNetwork},
    AIVec,
};

pub struct NetworkTemplate {
    pub n_inputs: usize,
    pub hidden_layers: Vec<usize>,
    pub n_outputs: usize,
}


pub struct TrainingData {
    n_inputs: usize,
    n_outputs: usize,
    length: usize,
    inputs: Vec<f64>,
    outputs: Vec<f64>,
}

impl TrainingData {
    pub fn new(n_inputs: usize, n_outputs: usize) -> Self {
        Self {
            n_inputs,
            n_outputs,
            length: 0,
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    pub fn add(&mut self, input: &[f64], output: &[f64]) {
        if input.len() == self.n_inputs && output.len() == self.n_outputs {
            input.iter().for_each(|i| self.inputs.push(*i));
            output.iter().for_each(|o| self.outputs.push(*o));
            self.length += 1;
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

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn to_string(&self) -> String {
        let mut out = String::new();

        for index in 0..self.length {
            write!(&mut out, "Set[{}]: I[", index).expect("Write Failure");

            let input_index_from = index * self.n_inputs;
            let input_index_to = input_index_from + self.n_inputs;

            self.inputs[input_index_from..input_index_to].into_iter()
                .for_each(|i| write!(&mut out, " {} ", i).expect("Write Failure") );

            write!(&mut out, "] O[").expect("Write Failure");

            let output_index_from = index * self.n_outputs;
            let output_index_to = output_index_from + self.n_outputs;

            self.outputs[output_index_from..output_index_to].into_iter()
                .for_each(|o| write!(&mut out, " {} ", o).expect("Write Failure") );

            writeln!(&mut out, "]").expect("Write Failure");
        }
        
        out
    }
}

pub struct Network {
    pub nn: NeuralNetwork,
    pub input: AIVec,
    pub output: AIVec,
}

impl Network {
    pub fn new(template: &NetworkTemplate) -> Self {
        let mut nb = NetworkBuilder::new(template.n_inputs);

        for hidden_size in &template.hidden_layers {
            nb.add_layer(*hidden_size, Activation::Sigmoid);
        }

        nb.add_layer(template.n_outputs, Activation::Sigmoid);

        Self {
            nn: nb.build().unwrap(),
            input: AIVec::zeros(template.n_inputs), 
            output: AIVec::zeros(template.n_outputs),
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