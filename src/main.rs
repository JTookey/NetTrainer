use eframe::egui;
use egui::plot::{Line, Plot, PlotPoint};

struct Netrainer {
    // Data
    training_error: Vec<[f64; 2]>,
}

impl eframe::App for Netrainer {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        
        egui::TopBottomPanel::new(egui::panel::TopBottomSide::Top, "Top Panel")
            .resizable(false)
            .min_height(20.0)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Run").clicked() {
                        
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
    let training_error = vec![[0.0, 0.0],[1.0, 20.0],[3.0, 10.0]];

    let gui = Box::new(Netrainer{
        training_error,
    });

    let options = eframe::NativeOptions::default();

    eframe::run_native(
        "NetTrainer", 
        options, 
        Box::new(|_| gui )
    );
}
