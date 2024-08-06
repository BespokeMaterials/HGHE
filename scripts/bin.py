import numpy as np

def read_npz_file(file_path, array_key=None):
    """
    Reads a .npz file and returns the requested array or a list of all arrays if no key is provided.

    Parameters:
    - file_path (str): Path to the .npz file.
    - array_key (str): Key of the specific array to retrieve (optional).

    Returns:
    - If array_key is provided: The numpy array associated with the key.
    - If no array_key is provided: A dictionary with all arrays in the file.
    """
    # Load the .npz file
    data = np.load(file_path, allow_pickle=True)

    # If a specific key is requested, return that array
    if array_key:
        if array_key in data.files:
            array = data[array_key]
            data.close()  # Close the file after accessing the array
            return array
        else:
            data.close()  # Close the file if key not found
            raise KeyError(f"Key '{array_key}' not found in the .npz file.")
    else:
        # If no key is provided, return all arrays in the file as a dictionary
        arrays_dict = {key: data[key] for key in data.files}
        data.close()  # Close the file after accessing the arrays
        return arrays_dict

# Example usage:

# To read and print all arrays in the file
arrays = read_npz_file('/Users/voicutomut/Downloads/graph_data_bilayer_MoS2/graph_data.npz')
print(arrays)

# To access a specific array by its key
specific_array = read_npz_file('example_file.npz', 'arr_0')
print(specific_array)
