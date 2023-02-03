use eframe::egui;
use egui::plot::{Line, Plot, PlotPoint};
use network::TrainingData;

mod network;
struct Netrainer {
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
                    if ui.button("Train").clicked() {
                        for i in 0..50000 {
                            
                            let (input, output) = self.training_data.get(i);

                            if let Ok(new_error) = self.network.train(&input, &output) {
                                self.training_error.push([self.training_error.len() as f64, new_error]);
                            }
                        }
                    }
                    if ui.button("Clear").clicked(){
                        
                    }
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            let line = Line::new(self.training_error.clone());
            Plot::new("Error Plot").view_aspect(2.0).show(ui, |plot_ui| plot_ui.line(line));
        });
    }

    fn on_exit(&mut self, _gl: Option<&eframe::glow::Context>) {}
}

fn main() {
    let training_data = network::TrainingData::new();
    let training_error = Vec::new();
    let network = network::Network::new();

    let gui = Box::new(Netrainer{
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
