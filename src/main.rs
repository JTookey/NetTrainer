use eframe::egui;
use egui::plot::{Line, Plot, PlotPoint};
use network::TrainingData;

enum GUI_State {
    New,
    Load,
    Details,
    Train,
    Test,
}

mod network;
struct Netrainer {
    // GUI
    current_state: GUI_State,

    // Data
    training_data: network::TrainingData,
    training_error: Vec<[f64; 2]>,

    // Neural Network
    network: network::Network,
}

impl eframe::App for Netrainer {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        
        egui::TopBottomPanel::new(egui::panel::TopBottomSide::Top, "Top Panel")
            .resizable(false)
            .min_height(20.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("New").clicked(){
                        self.current_state = GUI_State::New;
                    }
                    if ui.button("Load").clicked(){
                        self.current_state = GUI_State::Load;

                    } 
                    if ui.button("Details").clicked(){
                        self.current_state = GUI_State::Details;

                    } 
                    if ui.button("Train").clicked() {
                        self.current_state = GUI_State::Train;

                    }
                    if ui.button("Test").clicked(){
                        self.current_state = GUI_State::Test;

                    } 
                    if ui.button("Clear").clicked(){

                    }
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            // Render Panel based on the state of the GUI
            match self.current_state {
                GUI_State::New => {
                    ui.label("Create New AI Network");
                    ui.separator();

                },
                GUI_State::Load => {
                    ui.label("Load AI Network");
                    ui.separator();
                    

                },
                GUI_State::Details => {
                    ui.label("Details about current AI Network");
                    ui.separator();
                    ui.label(format!("Inputs: {}", self.network.nn.number_of_inputs()));
                    ui.label(format!("Layers: {}", self.network.nn.number_of_layers()));
                    ui.label(format!("Output: {}", self.network.nn.number_of_outputs()));
                    ui.separator();
                    ui.label(format!("Input Vec Shape: {:?}", self.network.input.strides()));
                },
                GUI_State::Train => {
                    ui.label("Train the current AI Network");
                    ui.separator();

                    if ui.button("Run Training").clicked() {
                        for i in 0..10000 {
                            let mut expectation = ai_core::AIVec::zeros(self.network.nn.number_of_outputs());

                            if let Ok(()) = self.training_data.get(i, &mut self.network.input, &mut expectation) {
                                
                                if let Ok(new_error) = self.network.train(&expectation) {
                                    self.training_error.push([self.training_error.len() as f64, new_error]);
                                }
                            }
                        }
                    }

                    let line = Line::new(self.training_error.clone());
                    Plot::new("Error Plot")
                        .view_aspect(2.0)
                        .show(ui, |plot_ui| 
                            plot_ui.line(line)
                        );
                },
                GUI_State::Test => {
                    ui.label("Test AI Network");
                    ui.separator();

                    // Inputs
                    for i in 0..self.network.nn.number_of_inputs() {
                        if ui.radio(self.network.input[i] == 1.0, format!("Input {}", i)).clicked() {
                            self.network.input[i] = 1.0 - self.network.input[i]; 
                        }
                    }
                    ui.separator();

                    // Process Buttons
                    if ui.button("Process").clicked() {
                        if let Ok(res) = self.network.process() {
                        }
                    }
                    ui.separator();

                    // Outputs
                    for i in 0..self.network.nn.number_of_outputs() {
                        ui.label(format!("Output {}: {}", i, self.network.output[i]));
                    }
                },
            }
        });
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {}
}

fn main() {
    let training_data = network::TrainingData::new();
    let training_error = Vec::new();
    let network = network::Network::new();

    let gui = Box::new(Netrainer{
        current_state: GUI_State::Train,
        training_data,
        training_error,
        network,
    });

    let options = eframe::NativeOptions::default();

    eframe::run_native(
        "NetTrainer", 
        options, 
        Box::new(|_| gui )
    );
}
