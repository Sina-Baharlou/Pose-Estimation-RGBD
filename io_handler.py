"""
3D pose estimation of an RGB-D camera using least squares technique
Created on Aug 2016
Updated on May 2019
By Sina M.Baharlou (Sina.baharlou@gmail.com)
Web page: www.sinabaharlou.com
"""

# -- Import required libraries --
import os


# -- Io Handler class --
class IoHandler:

    # -- Constructor --
    def __init__(self, path):
        self.__path = path  # -- working path
        self.__list = list()  # -- file list
        self.__current_index = 0  # -- current index

    # -- Clear file list --
    def clear_list(self):
        self.__list = list()
        self.__current_index = 0

    # -- Load all files --
    def load_files(self, extension, l_sorted=False):
        file_count = 0

        # -- Loop through all files --
        for file in os.listdir(self.__path):
            if file.endswith(extension):
                self.__list.append(file)
                file_count += 1

        # -- Sort the files if it's needed --
        if l_sorted:
            self.__list.sort()

        return file_count

    # -- Get the next file --
    def next_file(self, append_path=True):
        # -- Check if it has reached the end --
        if self.__current_index >= len(self.__list):
            return None

        # -- Get the filename --
        filename = self.__list[self.__current_index]

        # -- Append with path if it's needed --
        if append_path:
            filename = self.__path + filename

        # -- Go to the next file --
        self.__current_index += 1
        return filename

    # -- Get file at--
    def file_at(self, file_index, append_path=True):
        # -- Check if it's in the range --
        if file_index >= len(self.__list):
            return None

        # -- Get the filename --
        filename = self.__list[file_index]

        # -- Append with path if it's needed --
        if append_path:
            filename = self.__path + filename

        # -- Go to the next file --
        return filename

    # -- Get file list --
    def get_list(self):
        return self.__list
