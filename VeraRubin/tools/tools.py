# vera rubin v1.0
# tools.tools.py

def progressbar(current_value, total_value, bar_length=20, progress_char='#'): 
    """
    Display a progress bar in the console.
    
    Parameters
    ----------
    current_value : int
        Current progress value.
    total_value : int
        Total value for completion.
    bar_length : int, optional
        Length of the progress bar.
    progress_char : str, optional
        Character used to fill the progress bar.
    """
    if total_value == 0:
        print("Error: total_value cannot be 0")
        return
    
    # Calculate the percentage and progress
    percentage = int((current_value / total_value) * 100)
    progress = int((bar_length * current_value) / total_value)
    
    # Build the progress bar string
    loadbar = f"Progress: [{progress_char * progress}{'.' * (bar_length - progress)}] {percentage}%"
    
    # Print the progress bar (overwrite line until finished)
    end_char = '\r' if current_value < total_value else '\n'
    print(loadbar, end=end_char)