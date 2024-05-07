import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter.scrolledtext import ScrolledText
import tkinter.font as tkFont

import warnings
warnings.filterwarnings("ignore", category=FutureWarning) # remove future warnings for debugging

# Import custom modules
from vtt_formatting import format_VTT
from timeline_generator import create_timeline_figure
from stats_generator import create_stats_figure
from chunk_splitter import split_text_into_chunks
from summaries import abstractive_summarize_chunks, extractive_summarize_chunks, format_vtt_as_dialogue
from openai import summarize_text

# Window size
window_width = 800
window_height = 700

class VTTAnalyzer(tk.Tk):
    
    def __init__(self):
        """
        Initialize the VTTAnalyzer application with a specified window title, size, and a non-resizable configuration.
        It also initializes several attributes to None and sets up the GUI components of the application.
        """
        super().__init__()
        self.title("VTT File Analyzer")
        self.geometry(f"{window_width}x{window_height}")
        self.resizable(False, False)  # Make the window fixed in size
        self.setup_scrollable_window()  # Setup the main GUI components
        
        # Initialize multiple attributes to None for later use
        self.ab = None  # Abstractive summarization result placeholder
        self.ex = None  # Extractive summarization result placeholder
        self.timeline = None  # Timeline visualization placeholder
        self.stats = None  # Statistical data visualization placeholder
        self.chunks = None  # Text chunks for analysis placeholder
        self.canvas_widget = None  # Canvas widget for dynamic content display
        self.df = None  # DataFrame to hold VTT data, if applicable



    def setup_scrollable_window(self):
        """
        Setup the main window with a vertical scrollbar and a canvas for scrolling. This function
        initializes a canvas and a scrollbar and binds them to create a scrollable frame that fills
        the main window. It also sets up mouse wheel scrolling for the frame and initiates the widget
        creation process.
        """
        # Create a canvas with light blue background
        self.canvas = tk.Canvas(self, bg='lightblue')
        # Create a vertical scrollbar linked to the canvas's Y-axis scroll
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        # Frame that will hold the widgets
        self.scrollable_frame = ttk.Frame(self.canvas)

        # Bind frame configuration to dynamically update the canvas scroll region
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )

        # Create a window in the canvas that hosts the scrollable_frame
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        # Link scrollbar to canvas scroll
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Pack the canvas and scrollbar into the window
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Enable scrolling via the mouse wheel across the whole window
        self.bind_all("<MouseWheel>", self.on_mouse_wheel)

        # Call to create widgets in the scrollable area
        self.create_widgets()



    def create_widgets(self):
        """
        Create and pack the widgets in the scrollable frame.
        """
        
        custom_font = tkFont.Font(family="Helvetica", size=12) # Define the font
        
        self.upload_label = tk.Label(self.scrollable_frame, text="Please upload a VTT file:", # Upload label
                                        foreground="green", font=custom_font)
        self.upload_label.pack(pady=10, padx=10)

        self.upload_button = tk.Button(self.scrollable_frame, text="Upload File", command=self.open_file, # Upload button
                                        bg='#66c2a5', fg='white', font=custom_font)
        self.upload_button.pack(pady=10, padx=10)

        self.ex_analyze_button = tk.Button(self.scrollable_frame, text="Show Key Sentences", state='disabled', # Key Sentences button
                                        command=self.generate_ex_summary, bg='#8da0cb', fg='white', font=custom_font)
        self.ex_analyze_button.pack(pady=10, padx=10)

        self.ab_analyze_button = tk.Button(self.scrollable_frame, text="Show Summary", state='disabled', # Summary Button
                                        command=self.generate_ab_summary, bg='#8da0cb', fg='white', font=custom_font)
        self.ab_analyze_button.pack(pady=10, padx=10)

        self.summary_box = ScrolledText(self.scrollable_frame, height=10, width=80, # Text Box for summaries
                                        font=custom_font, bg='white', fg='black')
        self.summary_box.pack(pady=10, padx=10)

    def open_file(self):
        """
        Open a file dialog to allow the user to select a VTT file and then load its content into the application.
        Upon loading, the function will update the interface to reflect the loaded file and enable further analysis options.
        """
        # Open a file dialog to select a VTT file
        file_path = filedialog.askopenfilename(filetypes=[("VTT files", "*.vtt")])
        if file_path:
            try:
                # Attempt to format and load data from the selected VTT file
                self.df, self.formatted_content = format_VTT(file_path)
                # Enable analysis buttons once file is successfully loaded
                self.ex_analyze_button.config(state='normal')
                self.ab_analyze_button.config(state='normal')
                # Update the upload label to show the loaded file name
                self.upload_label.config(text=f"File loaded: {file_path.split('/')[-1]}")
                
                # Optional debugging message to confirm file load
                print("VTT file loaded successfully.")

                # Display visualizations and summaries based on the loaded data
                self.show_plot(create_timeline_figure)  # Visualize timeline data
                self.show_plot(create_stats_figure)  # Visualize statistical data
                # self.openai_summary()  # Generate summaries using OpenAI models

            except Exception as e:
                # Handle exceptions by showing an error message and updating the UI
                messagebox.showerror("Error", str(e))
                self.upload_label.config(text="Failed to load file.")


    def insert_text(self, text):
        """
        Insert text into the summary box widget, allowing modifications only during the update. The function also
        handles enabling buttons based on the content being displayed, depending on whether it matches predefined
        summaries for extractive or abstractive analysis results.

        Args:
            text (str): The text to be inserted into the summary box.
        """
        # Enable the text box for editing
        self.summary_box.configure(state='normal')
        # Clear existing content
        self.summary_box.delete('1.0', tk.END)
        # Insert new text
        self.summary_box.insert(tk.END, text)
        # Disable the text box to prevent user edits
        self.summary_box.configure(state='disabled')

        # Conditional UI update based on the text content
        match text:
            case self.ex:
                # Re-enable the button for extractive summary analysis if relevant
                self.ex_analyze_button.config(state='normal')
            case self.ab:
                # Re-enable the button for abstractive summary analysis if relevant
                self.ab_analyze_button.config(state='normal')


    def generate_ex_summary(self):
        """
        Generate an extractive summary from the loaded VTT content. This function checks if the summary has been
        previously generated. If not, it processes the content, generates the summary, and formats it. The summary
        is then displayed in the GUI.
        """
        # Check if the extractive summary has already been generated
        if self.ex is None:
            # Split the text into chunks if not already done
            if self.chunks is None:
                self.chunks = split_text_into_chunks(self.formatted_content)
            # Generate an extractive summary from the chunks
            final_summary = extractive_summarize_chunks(self.chunks)
            # Format the summary for display
            self.ex = format_vtt_as_dialogue(final_summary)
            # Optional debugging message to confirm the extraction
            print("Key Sentences Extracted")

        # Display the extractive summary in the GUI
        self.insert_text(self.ex)

    def generate_ab_summary(self):
        """
        Generate an abstractive summary from the loaded VTT content. This function checks if the summary has been
        previously generated. If not, it processes the content and generates the summary. The summary is then displayed
        in the GUI.
        """
        # Check if the abstractive summary has already been generated
        if self.ab is None:
            # Split the text into chunks if not already done
            if self.chunks is None:
                self.chunks = split_text_into_chunks(self.formatted_content)
            # Generate an abstractive summary from the chunks
            self.ab = abstractive_summarize_chunks(self.chunks)
            # Optional debugging message to confirm the generation
            print("Summary Generated")

        # Display the abstractive summary in the GUI
        self.insert_text(self.ab)


    def openai_summary(self):
        """
        Generate a summary using an OpenAI model and print it.
        """
        # Generate a summary for the formatted VTT content
        summary = summarize_text(self.formatted_content)
        # Print the generated summary to the console
        print(summary)


    def show_plot(self, plot_function):
        """
        Display a matplotlib plot in the GUI. This method generates a plot using a provided function and
        data, then integrates it into the scrollable canvas area of the interface.

        Args:
            plot_function (function): A function that takes a DataFrame and returns a matplotlib figure.
        """
        # Generate a plot with the provided function using the loaded data
        fig = plot_function(self.df)
        # Close the figure to free up memory resources
        plt.close(fig)

        # Embed the figure into the tkinter canvas using FigureCanvasTkAgg
        self.canvas_widget = FigureCanvasTkAgg(fig, master=self.scrollable_frame)
        self.canvas_widget.draw()
        self.canvas_widget.get_tk_widget().pack()
        # Update the scroll region to include the new plot
        self.update_scroll_region()

    def on_mouse_wheel(self, event):
        """
        Handle mouse wheel events to scroll the canvas vertically. This method adjusts the vertical
        view of the canvas based on the mouse wheel movement.

        Args:
            event: An event object containing information about the mouse wheel action.
        """
        # Determine the direction of the scroll (up or down)
        if event.num == 4 or event.delta > 0:  # Scroll up
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5 or event.delta < 0:  # Scroll down
            self.canvas.yview_scroll(1, "units")

    def update_scroll_region(self):
        """
        Update the scroll region of the canvas. This method is called after adding or resizing
        content within the scrollable area to ensure that the entire content is accessible
        through scrolling.
        """
        # Adjust the scroll region to fit the new size of the content
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))



if __name__ == "__main__":
    # Create an instance of the VTTAnalyzer application
    app = VTTAnalyzer()
    # Start the application's main event loop, waiting for user interaction
    app.mainloop()

