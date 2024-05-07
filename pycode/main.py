import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter.scrolledtext import ScrolledText
import tkinter.font as tkFont

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Import your custom modules
from vtt_formatting import format_VTT
from timeline_generator import create_timeline_figure
from stats_generator import create_stats_figure
from chunk_splitter import split_text_into_chunks
from summaries import abstractive_summarize_chunks, extractive_summarize_chunks, format_vtt_as_dialogue

# Window size
window_width = 800
window_height = 700

class VTTAnalyzer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VTT File Analyzer")
        self.geometry(f"{window_width}x{window_height}") 
        self.setup_scrollable_window()     
        self.ab, self.ex, self.timeline, self.stats, self.chunks, self.canvas_widget, self.df = [None]*7


    def setup_scrollable_window(self):
        self.canvas = tk.Canvas(self, bg='lightblue')
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.bind_all("<MouseWheel>", self.on_mouse_wheel)  # Windows and macOS
        self.maxsize(width=self.winfo_screenwidth(), height=self.winfo_screenheight() - 100)
        self.create_widgets()

    def create_widgets(self):
        custom_font = tkFont.Font(family="Helvetica", size=12)
        
        self.upload_label = tk.Label(self.scrollable_frame, text="Please upload a VTT file:", 
                                        foreground="green", font=custom_font)
        self.upload_label.pack(pady=10, padx=10)

        self.upload_button = tk.Button(self.scrollable_frame, text="Upload File", command=self.open_file,
                                        bg='#66c2a5', fg='white', font=custom_font)
        self.upload_button.pack(pady=10, padx=10)

        self.ex_analyze_button = tk.Button(self.scrollable_frame, text="Show Key Sentences", state='disabled',
                                        command=self.generate_ex_summary, bg='#8da0cb', fg='white', font=custom_font)
        self.ex_analyze_button.pack(pady=10, padx=10)

        self.ab_analyze_button = tk.Button(self.scrollable_frame, text="Show Summary", state='disabled',
                                        command=self.generate_ab_summary, bg='#8da0cb', fg='white', font=custom_font)
        self.ab_analyze_button.pack(pady=10, padx=10)

        # self.plot1_button = tk.Button(self.scrollable_frame, text="Show Timeline", state='disabled',
        #                                 command=lambda: self.show_plot(create_timeline_figure),
        #                                 bg='maroon', fg='white', font=custom_font)
        # self.plot1_button.pack(pady=10, padx=10)

        # self.plot2_button = tk.Button(self.scrollable_frame, text="Show Speaker Stats", state='disabled',
        #                                 command=lambda: self.show_plot(create_stats_figure),
        #                                 bg='purple', fg='white', font=custom_font)
        # self.plot2_button.pack(pady=10, padx=10)

        self.summary_box = ScrolledText(self.scrollable_frame, height=10, width=80,
                                        font=custom_font, bg='white', fg='black')
        self.summary_box.pack(pady=10, padx=10)

    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("VTT files", "*.vtt")])
        if file_path:
            try:
                self.df, self.formatted_content = format_VTT(file_path)
                self.ex_analyze_button.config(state='normal')
                self.ab_analyze_button.config(state='normal')
                # self.plot1_button.config(state='normal')
                # self.plot2_button.config(state='normal')
                self.upload_label.config(text=f"File loaded: {file_path.split('/')[-1]}")
                print()
                print("vtt loaded")
                self.show_plot(create_timeline_figure)
                self.show_plot(create_stats_figure)
            except Exception as e:
                messagebox.showerror("Error", str(e))
                self.upload_label.config(text="Failed to load file.")

    def insert_text(self, text):
        self.summary_box.configure(state='normal')
        self.summary_box.delete('1.0', tk.END)
        self.summary_box.insert(tk.END, text)
        self.summary_box.configure(state='disabled')
        match text:
            case self.ex:
                self.ex_analyze_button.config(state='normal')
            case self.ab:
                self.ab_analyze_button.config(state='normal')

    def generate_ex_summary(self):
        if self.ex == None : 
            if self.chunks == None :
                self.chunks = split_text_into_chunks(self.formatted_content)
            final_summary = extractive_summarize_chunks(self.chunks)
            self.ex = format_vtt_as_dialogue(final_summary)
            print()
            print("Key Sentences Extracted")

        self.insert_text(self.ex)
        # self.summary_box.configure(state='normal')
        # self.summary_box.delete('1.0', tk.END)
        # self.summary_box.insert(tk.END, self.ex)
        # self.summary_box.configure(state='disabled')
        # self.ex_analyze_button.config(state='normal')
    
    def generate_ab_summary(self):
        if self.ab == None : 
            if self.chunks == None :
                self.chunks = split_text_into_chunks(self.formatted_content)
            self.ab = abstractive_summarize_chunks(self.chunks)
            print()
            print("Summary Generated")

        self.insert_text(self.ab)
        # self.summary_box.configure(state='normal')
        # self.summary_box.delete('1.0', tk.END)
        # self.summary_box.insert(tk.END, self.ab)
        # self.summary_box.configure(state='disabled')
        # self.ab_analyze_button.config(state='normal')

    def show_plot(self, plot_function):
        fig = plot_function(self.df)
        plt.close(fig)  # Close the figure to prevent memory leak
        # if self.canvas_widget:
        #     self.canvas_widget.get_tk_widget().destroy()  # Destroy the previous canvas if exists

        self.canvas_widget = FigureCanvasTkAgg(fig, master=self.scrollable_frame)
        self.canvas_widget.draw()
        self.canvas_widget.get_tk_widget().pack()

    # def show_plot1(self):
    #     self.show_plot(create_timeline_figure(self.df))

    # def show_plot2(self):
    #     self.show_plot(create_stats_figure(self.df))

    def on_mouse_wheel(self, event):
        if event.num == 4 or event.delta > 0:  # Scroll up (Linux or Windows)
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0:  # Scroll down (Linux or Windows)
            self.canvas.yview_scroll(1, "units")

if __name__ == "__main__":
    app = VTTAnalyzer()
    app.mainloop()
