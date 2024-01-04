import customtkinter
import logging

import MatchingModels
from widgets import ResultView, PopUpMessage, FloderSelection


logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%d/%m/%y %I:%M:%S %p",
    filename="logfile.log",
    encoding="utf-8",
    level=logging.DEBUG,
)

class MainApp(customtkinter.CTk):

    def __init__(self):
        super().__init__()

        self.title("Demo")

        self.geometry("1200*500")

        self.left_bar = customtkinter.CTkFrame(
            master=self, width=60, height=250, border_width=1
        )
        self.tabview = customtkinter.CTkTabview(
            master=self, width=300, height=250, border_width=1
        )
        self.tabview.add("Data")
        self.tabview.add("Result")
        self.tabview.set("Data")

        self.rowconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.left_bar.grid(row=0, column=0, sticky="nesw")
        self.tabview.grid(row=0, column=1, sticky="nesw")
        
        self.model: MatchingModels.FeaturesExtractor = None
        self.inferencer: MatchingModels.Inferencer = None
        # In left _bar
        self.closebutton = customtkinter.CTkButton(
            master=self.left_bar,
            command=self.close_window,
            text="close",
            fg_color="Red",
            anchor="center",
            border_width=1,
        )
        self.model_combobox = customtkinter.CTkComboBox(
            master=self.left_bar,
            values=list(MatchingModels.models_list.keys()),
            command=self.model_combobox_callback,
        )
        self.inference_button = customtkinter.CTkButton(
            master=self.left_bar,
            command=self.inference,
            text="inference",
            border_width=1,
        )
        self.trainning_cbox = customtkinter.CTkCheckBox(
            master=self.left_bar,
            text = "train",
            border_width=1,
        )

        self.left_bar.columnconfigure(0, weight=1)
        self.left_bar.rowconfigure((0, 1, 2,3), weight=1)
        self.closebutton.grid(row=0, sticky="n", padx=10, pady=10)
        self.model_combobox.grid(row=1)
        self.inference_button.grid(row=2)
        self.trainning_cbox.grid(row=3)

        # In Right tab views tab(Data)
        self.target_frame = FloderSelection(
            master=self.tabview.tab("Data"), text="Target", image_size = (100,100), border_width=1, 
        )
        self.query_frame = FloderSelection(
            master=self.tabview.tab("Data"), text="Query", image_size = (100,100), border_width=1
        )

        self.tabview.tab("Data").rowconfigure((0, 1), weight=1)
        self.tabview.tab("Data").columnconfigure(0, weight=1)
        self.target_frame.grid(row=0, column=0, sticky="nesw", pady=5)
        self.query_frame.grid(row=1, column=0, sticky="nesw", pady=5)

        # In Right tab views tab(Result)
        self.result_window = ResultView(master=self.tabview.tab("Result"), image_size = (200,200),width=300, height=200, corner_radius=0, fg_color="transparent")

        self.tabview.tab("Result").rowconfigure(0, weight=1)
        self.tabview.tab("Result").columnconfigure(0, weight=1)
        self.result_window.grid(row=0, column=0, sticky = "nesw")

    def close_window(self) -> None:
        self.destroy()  # close the main apps window
        self.quit()  # quit python runtime
    

    def model_combobox_callback(self, choice) -> None:
        self.model = MatchingModels.FeaturesExtractor(
            model_constructor = MatchingModels.models_list[choice],
            weights="DEFAULT",
            frozen=False,
        )
        logging.debug(f"Model {self.model.name} created ")

    def inference(self) -> None:
        try:
            if self.query_frame.path is None or self.target_frame.path is None:
                raise TypeError("The paths given is None")
            
            if self.inferencer is None or self.inferencer.model != self.model:
                self.inferencer = MatchingModels.Inferencer(self.model, self.query_frame.path, self.target_frame.path)
                logging.debug(f"New Inferencer {self.inferencer.model.name} created ")
            
            logging.debug(f"generating resut with {self.inferencer.model.name} for q-path{self.query_frame.path}, t-path{self.target_frame.path}")

            if self.trainning_cbox.get():
                self.inferencer.train_model()

            self.result_window.generate_results(self.inferencer, k=5)
            self.tabview.set("Result")
            
        except AttributeError:
            pop_up = PopUpMessage(master=self, text="Selscte a model first")
            pop_up.focus()
        except TypeError:
            pop_up = PopUpMessage(master=self, text="Select path first")
            pop_up.focus()


if __name__ == "__main__":
    app = MainApp()
    logging.debug("Opened App")
    app.mainloop()