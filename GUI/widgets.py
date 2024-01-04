from typing import Literal, Optional, Tuple, Union
from typing_extensions import Literal
import customtkinter

import tkinter as tk
import logging

from pathlib import Path
from PIL import Image
from collections import namedtuple

from customtkinter.windows.widgets.font import CTkFont

import MatchingModels
import torch

ImageSize = namedtuple("ImageSize", "width height")

class PopUpMessage(customtkinter.CTkToplevel):
    def __init__(self, text, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.geometry("400x200")
        self.label = customtkinter.CTkLabel(master=self, text=text, width=400, height=200)
        self.label.grid(row=0, column=0, sticky = "nesw")

class FloderSelection(customtkinter.CTkFrame):
    IMAGE_SIZE = ImageSize(50, 50)
    def __init__(self, master, text:str, image_size: tuple = None, width=140, height=28, **kwarg):
        super().__init__(master, width, height, **kwarg)
        
        self.image_size = ImageSize(*image_size) if image_size is not None else self.__class__.IMAGE_SIZE
        
        self.label = customtkinter.CTkLabel(master=self, text=text, anchor="center")
        self.file_box = customtkinter.CTkEntry(
            master=self, height=28, placeholder_text="Entry path here"
        )
        self.ask_file_button = customtkinter.CTkButton(
            master=self,
            text="Select Folder",
            width=84,
            command=self.ask_path,
            anchor="center",
        )

        # DLLM the tk.Text is using diff height unit, so set to 5
        self.preview_win = tk.Text(
            master=self, wrap="word", bg="black", height=5
        )  # use tk text wdiget for autop wrapping

        self.rowconfigure(1, weight=1)
        self.columnconfigure(1, weight=1)
        self.preview_win.grid(row=1, column=0, columnspan=3, sticky="nesw")
        self.label.grid(row=0, column=0, sticky="e", padx=5, pady=5)
        self.file_box.grid(row=0, column=1, sticky="nesw")
        self.ask_file_button.grid(row=0, column=2, padx=5, pady=5)

    @property
    def path(self) -> Path:
        "the path of selected floder"
        path = self.file_box.get()
        return Path(path) if path != "" else None

    def ask_path(self) -> None:
        if self.file_box.get() != "":
            self.clear_path()
        dict_name = tk.filedialog.askdirectory(mustexist=True)
        self.file_box.insert("0", dict_name)
        logging.debug(f"Selected file {self.path}")
        self.create_preview()

    def clear_path(self) -> None:
        self.file_box.delete(0, tk.END)
        self.clear_preview()

    def create_preview(self) -> None:

        for file in self.path.glob("*"):
            label = customtkinter.CTkLabel(
                master=self.preview_win,
                width=self.image_size.width,
                height=self.image_size.height,
                text="",
                image=customtkinter.CTkImage(
                    light_image=Image.open(file), size=self.image_size
                ),
            )
            self.preview_win.window_create("end", window=label, padx=10, pady=10)

    def clear_preview(self) -> None:
        self.preview_win.delete("1.0", tk.END)

class ResultView(customtkinter.CTkScrollableFrame):
    IMAGE_SIZE = ImageSize(100, 100)
    
    def __init__(self, master, image_size:tuple = None, **kwarg):
        super().__init__(master, **kwarg)
        
        self.image_size = ImageSize(*image_size) if image_size is not None else self.__class__.IMAGE_SIZE

        self.results = []
        self.columnconfigure(0, weight=1)

    def generate_results(self, inferencer: MatchingModels.Inferencer, k=5):
        
        result = inferencer.find_k_most_similar(k)

        for i, (query_file, match_pairs) in enumerate(result):

            result_row = customtkinter.CTkFrame(master=self, border_width=1)
            self.results.append(result_row)

            query_photo = customtkinter.CTkLabel(
                master=result_row,
                width=self.image_size.width,
                height=self.image_size.height,
                text="",
                image=customtkinter.CTkImage(
                    light_image=Image.open(query_file), size = self.image_size
                ),
            )
            query_label = customtkinter.CTkLabel(master=result_row,width=self.image_size.width,height=28,text="Input")

            self.rowconfigure(i, weight=0)
            result_row.grid(row=i, column=0, sticky = "nesw")
            result_row.columnconfigure(0, weight=1)
            result_row.rowconfigure(0, weight=1)
            query_photo.grid(row=0, column=0, sticky = "n")
            query_label.grid(row=1, column=0, sticky = "n")

            for j, (sorce, target_files) in enumerate(match_pairs,1):
                result_photo = customtkinter.CTkLabel(
                    master=result_row,
                    width=self.image_size.width,
                    height=self.image_size.height,
                    text="",
                    image=customtkinter.CTkImage(
                        light_image=Image.open(target_files), size=self.image_size
                    ),
                )
                result_label = customtkinter.CTkLabel(master=result_row,width=self.image_size.width,height=28,text=f"{sorce*100 :.2f}")
                result_row.columnconfigure(j, weight=1)
                result_photo.grid(row=0, column = j, sticky = "n")
                result_label.grid(row=1, column = j, sticky = "n")

            logging.debug(f"result done for {i}-th row for query")
    
    def clear_result(self):
        for frame in self.results:
            frame.destroy()
