import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter.scrolledtext import ScrolledText
import tkinter.font as tkFont

import warnings
warnings.filterwarnings("ignore", category=FutureWarning) # remove future warnings for debugging

# Import custom modules
# from packages.vtt_formatting import format_VTT
# from packages.timeline_generator import create_timeline_figure
# from packages.stats_generator import create_stats_figure
# from packages.chunk_splitter import split_text_into_chunks
# from packages.summaries import abstractive_summarize_chunks, extractive_summarize_chunks, format_vtt_as_dialogue
# from packages.openai import summarize_text, utility_text
# from packages.sentiment import sentiment

# Window size
window_width = 1400
window_height = 800

window_min_width = 800
window_min_height = 600

class VTTAnalyzer(tk.Tk):


    def __init__(self):
        super().__init__()
        
        self.title("VTT File Analyzer")
        self.geometry(f"{window_width}x{window_height}")
        # self.resizable(False, False)  # Make the window fixed in size
        self.minsize(window_min_width, window_min_height)  # Set the minimum size of the window
        self.custom_font = tkFont.Font(family="Helvetica", size=12)  # Define the font

        self.style = ttk.Style()
        self.style.configure('TButton', background='#8da0cb', foreground='blue', font=self.custom_font)
        self.style.configure('TLabel', font=self.custom_font, foreground='darkgrey')
        self.style.configure('TScale', background='#8da0cb')

        print("Window Initialized")

        self.ab = None  # Abstractive summarization result placeholder
        self.ex = None  # Extractive summarization result placeholder
        self.ai = None  # Summary generated by openAI api
        self.timeline = None  # Timeline visualization placeholder
        self.stats = None  # Statistical data visualization placeholder
        self.chunks = None  # Text chunks for analysis placeholder
        self.canvas_widget = None  # Canvas widget for dynamic content display
        self.df = None  # DataFrame to hold VTT data, if applicable

        self.setup_scrollable_window()  # Setup the main GUI components

        print("Initialized")



    def setup_scrollable_window(self):
        
        ##############################################################################################################   RED
        # Create a container frame
        self.container = tk.Frame(self, bg='red')
        self.container.grid(sticky="nsew")

        # Configure the main window grid to expand properly
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        # Configure the container grid to expand properly
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(1, weight=1)

        ##############################################################################################################   YELLOW
        self.button_frame = tk.Frame(self.container, bg='yellow')
        self.button_frame.grid(row=0, column=0, sticky="ns", pady=10)

        ##############################################################################################################   LIGHTBLUE
        self.content_frame = tk.Frame(self.container, bg='lightblue')
        self.content_frame.grid(row=0, column=1, sticky="news", pady=10, columnspan=4)

        ##############################################################################################################   ORANGE
        self.upload_frame = tk.Frame(self.content_frame, bg='orange')
        self.upload_frame.grid(row=0, column=0, sticky="n", pady=10)

        ##############################################################################################################   PURPLE
        # self.canvas_frame = tk.Frame(self.content_frame, bg='purple')
        # self.canvas_frame.grid(row=1, column=0, sticky="news", pady=10, columnspan=2)

        ##############################################################################################################   LIGHTGREEN
        self.scrollable_frame = tk.Frame(self.content_frame, bg='purple')
        self.scrollable_frame.grid(row=1, column=0, sticky="n", pady=10)
        
        self.scrollbar = ttk.Scrollbar(self.content_frame)
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        # self.scrollbar.config(command=self.content_frame.yview)

        
        
        # # Create a canvas and place it in the canvas_frame
        # self.canvas = tk.Canvas(self.canvas_frame, bg='lightgreen')
        # self.canvas.grid(row=0, column=0, sticky="ensw")

        # # Create a vertical scrollbar linked to the canvas
        # self.scrollbar = ttk.Scrollbar(self.canvas_frame, orient="vertical", command=self.canvas.yview)
        # self.scrollbar.grid(row=0, column=1, sticky="ns")

        # # Configure the canvas to use the scrollbar
        # self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # # Create a frame inside the canvas
        # self.scrollable_frame = ttk.Frame(self.canvas_frame)
        # self.scrollable_frame.grid(row=0, column=0, sticky="n", pady=10)

        # # Create a window inside the canvas
        # self.window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        # # Ensure the scroll region is updated whenever the size of the frame changes
        # self.scrollable_frame.bind(
        #     "<Configure>",
        #     lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        # )

        # # Ensure the scrollable frame width matches the canvas frame width
        # self.canvas.bind("<Configure>", self.on_canvas_configure)

        # # Make the content_frame expandable
        # self.content_frame.grid_columnconfigure(0, weight=1)
        # self.content_frame.grid_rowconfigure(1, weight=1)

        self.create_buttons()
        self.create_widgets()


    def create_buttons(self):
        button_options = {
            'style': 'TButton',
            'padding': 10,
            'width': 25
        }

        self.ex_analyze_button = ttk.Button(self.button_frame, text="Show Key Sentences", state='disabled', 
                                            **button_options)
        self.ex_analyze_button.grid(row=0, column=0, pady=10, padx=10, sticky="ew")

        self.ab_analyze_button = ttk.Button(self.button_frame, text="Show Abstractive Summary", state='disabled', 
                                            **button_options)
        self.ab_analyze_button.grid(row=1, column=0, pady=10, padx=10, sticky="ew")

        self.ai_analyze_button = ttk.Button(self.button_frame, text="Show openAI Summary", state='disabled', 
                                            **button_options)
        self.ai_analyze_button.grid(row=2, column=0, pady=10, padx=10, sticky="ew")

        self.transcript_button = ttk.Button(self.button_frame, text="Show full transcript", state='disabled', 
                                            **button_options)
        self.transcript_button.grid(row=3, column=0, pady=10, padx=10, sticky="ew")

        self.sentiment_button = ttk.Button(self.button_frame, text="Show sentiment analysis", state='disabled', 
                                           **button_options)
        self.sentiment_button.grid(row=4, column=0, pady=10, padx=10, sticky="ew")
        
        self.ai_utility_button = ttk.Button(self.button_frame, text="Show Key Decisions", state='disabled', 
                                            **button_options)
        self.ai_utility_button.grid(row=5, column=0, pady=10, padx=10, sticky="ew")
        
        # Create an Entry widget
        self.prompt = tk.Entry(self.button_frame, state='disabled')
        self.prompt.grid(row=6, column=0, pady=10, padx=10, sticky="ew")
        
        # Create a Button to trigger the get_text function
        self.prompt_button = tk.Button(self.button_frame, text="Get Text", command=self.get_prompt)
        self.prompt_button.grid(row=7, column=0, pady=10, padx=10, sticky="ew")
        
    def create_widgets(self):
        self.upload_label = tk.Label(self.upload_frame, text="Please upload a VTT file:", 
                                    foreground="green", font=self.custom_font)
        self.upload_label.grid(row=0, column=0, pady=10, padx=10, sticky="n")

        self.upload_button = tk.Button(self.upload_frame, text="Upload File",
                                    bg='#66c2a5', fg='white', font=self.custom_font)
        self.upload_button.grid(row=1, column=0, pady=10, padx=10, sticky="n")

        self.summary_label = ttk.Label(self.scrollable_frame, text="Select Summary Length", style='TLabel')
        self.summary_label.grid(row=3, column=0, pady=[20, 0], padx=10, sticky="n")

        self.slider = ttk.Scale(self.scrollable_frame, from_=50, to_=1000, orient=tk.HORIZONTAL, length=250)
        self.slider.set(100)
        self.slider.grid(row=4, column=0, pady=10, padx=10, sticky="n")

        self.value_label = ttk.Label(self.scrollable_frame, text=f"Total words set to: {int(self.slider.get())}", style='TLabel')
        self.value_label.grid(row=5, column=0, pady=10, padx=10, sticky="n")

        # self.slider.bind("<Motion>", self.update_label)

        self.summary_box = ScrolledText(self.scrollable_frame, height=10, width=80, 
                                        font=self.custom_font, bg='white', fg='black')
        self.summary_box.grid(row=6, column=0, pady=10, padx=10, sticky="new")

        self.summary_label = ttk.Label(self.scrollable_frame, text="Summary", style='TLabel')
        self.summary_label.grid(row=7, column=0, pady=[20, 0], padx=10, sticky="n")

        self.summary_box = ScrolledText(self.scrollable_frame, height=10, width=80, 
                                        font=self.custom_font, bg='white', fg='black')
        self.summary_box.grid(row=8, column=0, pady=10, padx=10, sticky="n")

        # self.summary_box.bind("<MouseWheel>", self.on_mouse_wheel_textbox)
        # self.bind_all("<MouseWheel>", self.on_mouse_wheel_window)

        
        # # Create an Entry widget for the search term
        # self.search_entry = tk.Entry(self.scrollable_frame)
        # self.search_entry.pack(side='top', fill='x')

        # # Bind the Enter key to the start_search method
        # self.search_entry.bind('<Return>', self.start_search)

        # # Create a Search button
        # search_button = tk.Button(self.scrollable_frame, text='Search', command=self.start_search)
        # search_button.pack(side='top')
        
        print(), print("Widgets Created")
        
        # Get the text from the Entry widget
        
        
        
    def get_prompt(self):
        text = self.prompt.get()
        print("You entered:", text)
    
        
    def on_canvas_configure(self, event):
        # Update the scrollable_frame width to match the canvas width
        canvas_width = event.width
        self.canvas.itemconfig(self.window, width=canvas_width)