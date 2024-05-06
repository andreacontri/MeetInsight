import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import your custom modules
from vtt_formatting import format_VTT
from timeline_generator import create_timeline_figure
from stats_generator import create_stats_figure
from chunk_splitter import split_text_into_chunks
from summaries import abstractive_summarize_chunks, extractive_summarize_chunks, format_vtt_as_dialogue

# Window size
window_width = 900
window_height = 700

class VTTAnalyzer(tk.Tk):
    """
    The main application class.
    """

    def __init__(self):
        super().__init__()
        self.title("VTT File Analyzer")
        self.geometry(f"{window_width}x{window_height}")
        # self.resizable(False, False)
        
        self.graph_visibility = {} # Dic that keeps track of the toggle 
        self.create_widgets()

    def create_widgets(self):
        """
        Create the widgets for the application.
        """
        # Create a scrollable frame
        scrollable_frame = create_scrollable_frame(self)

        # Create the upload frame
        upload_frame = ttk.Frame(scrollable_frame)
        upload_frame.pack(fill=tk.X, padx=20, pady=20)

        # Status label for confirming file upload
        self.status_label = ttk.Label(upload_frame, text="", foreground="green")
        self.status_label.pack(side=tk.TOP, pady=10)

        # Create a button to upload files
        upload_btn = ttk.Button(upload_frame, text="Upload VTT File", command=self.upload_file)
        upload_btn.pack(side=tk.TOP, pady=20)

        # Create the graph and button frame
        graph_and_button_frame = ttk.Frame(scrollable_frame)
        graph_and_button_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Create the button frame
        self.button_frame = ttk.Frame(graph_and_button_frame)
        self.button_frame.pack(side=tk.TOP, fill=tk.Y, padx=20, pady=20)

        # Create the graph frame
        self.graph_frame = ttk.Frame(graph_and_button_frame)
        self.graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Create the output frame
        output_frame = ttk.Frame(scrollable_frame, height=200)
        # output_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=20, pady=20, expand=False)
        output_frame.pack_propagate(False)  # Prevent the frame from resizing to fit its contents
        output_frame.pack(side="bottom", fill="x", expand=False)  # Adjust these as needed

        # Create a text widget to display the final output
        self.output_text = tk.Text(output_frame, height=20, width=90, state='disabled')
        self.output_text.pack(side=tk.TOP, pady=10)
        
        scrollable_frame.pack(fill=tk.BOTH, expand=True)

    def upload_file(self):
        """
        Open a file dialog to select a VTT file, process the file, and generate graphs and summary.
        """
        # Open a dialog to choose the file
        file_path = filedialog.askopenfilename(filetypes=[("VTT files", "*.vtt")])
        if file_path:
            try:
                # Validate the file
                if not self.validate_file(file_path):
                    return

                # Show a loading screen
                self.show_loading_screen()

                # Format the transcript and generate the data, save the dataframe in df, the text version in formatted_content
                df, formatted_content = format_VTT(file_path)

                # Generate the plot figures and generate the buttons to show them
                self.figures = [create_timeline_figure(df), create_stats_figure(df)]
                self.create_graph_buttons(self.figures)

                # Generate and show the summary
                self.generate_summary(formatted_content)

                # Show filename as confirmation
                self.update_status_label(f"File loaded: {file_path.split('/')[-1]}")

            except Exception as e:
                messagebox.showerror("Error", str(e))
                self.update_status_label("Failed to load file.")
            finally:
                # Hide the loading screen
                self.hide_loading_screen()

    def validate_file(self, file_path):
        """
        Validate the selected file.

        Args:
            file_path (str): The path to the selected file.

        Returns:
            bool: True if the file is valid, False otherwise.
        """
        # Add your file validation logic here
        # For example, you can check if the file extension is .vtt
        if not file_path.endswith(".vtt"):
            messagebox.showerror("Invalid File", "Please select a valid VTT file.")
            return False
        return True

    def generate_summary(self, vtt):
        """
        Generate a summary from the formatted VTT content and display it in the output text widget.

        Args:
            vtt (str): The formatted VTT content.
        """
        try:
            chunks = split_text_into_chunks(vtt)
            final_summary = extractive_summarize_chunks(chunks)
            final_output = format_vtt_as_dialogue(final_summary)

            self.output_text.configure(state='normal')
            self.output_text.delete('1.0', tk.END)  # Clear existing text
            self.output_text.insert(tk.END, final_output)  # Insert new text
            self.output_text.configure(state='disabled')
        except Exception as e:
            messagebox.showerror("Error", "Failed to generate or display summary: " + str(e))
            self.output_text.configure(state='disabled')


    def create_graph_buttons(self, figures):
        """
        Create buttons to display the generated graphs.

        Args:
            figures (list): A list of matplotlib figure objects.
        """
        # Clear previous buttons if any
        for widget in self.button_frame.winfo_children():
            widget.destroy()

        # Create a new button for each figure
        graph_names = ["Timeline Graph", "Stats Graph"]
        self.graph_buttons = []
        for fig, name in zip(figures, graph_names):
            btn = ttk.Button(self.button_frame, text=f"Show {name}", command=lambda f=fig: self.toggle_graph(f))
            btn.pack(side=tk.TOP, pady=5)
            self.graph_buttons.append(btn)

    def toggle_graph(self, fig):
        """
        Toggle the display of the selected graph.

        Args:
            fig (matplotlib.figure.Figure): The matplotlib figure object to display or hide.
        """
        # Check if the graph is currently visible
        is_visible = self.graph_visibility.get(fig, False)

        # Clear the graph frame if the graph is visible
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        # Toggle the visibility
        if not is_visible:
            # If the graph was not visible, display it
            canvas = FigureCanvasTkAgg(fig, master=self.graph_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.graph_visibility[fig] = True  # Update visibility state to True
        else:
            # If the graph was visible, it has been hidden now
            self.graph_visibility[fig] = False  # Update visibility state to False

    def update_status_label(self, message):
        """
        Update the status label with the given message.

        Args:
            message (str): The message to display in the status label.
        """
        self.status_label.config(text=message)

    def get_canvas(self, fig):
        """
        Get the canvas associated with the given figure, if it exists.

        Args:
            fig (matplotlib.figure.Figure): The matplotlib figure object.

        Returns:
            tk.Canvas or None: The canvas associated with the figure, or None if not found.
        """
        for widget in self.graph_frame.winfo_children():
            if isinstance(widget, FigureCanvasTkAgg):
                if widget.figure == fig:
                    return widget.get_tk_widget()
        return None

    def show_loading_screen(self):
        """
        Show a loading screen while processing the file.
        """
        # Create a loading screen (e.g., a progress bar or a spinning animation)
        self.loading_screen = ttk.Label(self, text="Processing file... Please wait.")
        self.loading_screen.pack(pady=20)

    def hide_loading_screen(self):
        """
        Hide the loading screen after processing the file.
        """
        self.loading_screen.pack_forget()


def create_scrollable_frame(parent):
    """
    Create a scrollable frame inside the parent widget.

    Args:
        parent (tk.Widget): The parent widget for the scrollable frame.

    Returns:
        tk.Frame: The scrollable frame.
    """
    canvas = tk.Canvas(parent)
    scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)

    scrollbar.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    frame = ttk.Frame(canvas)
    canvas_frame = canvas.create_window((0, 0), window=frame, anchor="nw")

    def on_frame_configure(event):
        canvas.configure(scrollregion=canvas.bbox("all"))

    frame.bind("<Configure>", on_frame_configure)

    # Bind mouse wheel event to canvas for scrolling
    parent.bind("<MouseWheel>", lambda event: canvas.yview_scroll(int(-1*(event.delta/120)), "units"))

    return frame




if __name__ == "__main__":
    app = VTTAnalyzer()
    app.mainloop()