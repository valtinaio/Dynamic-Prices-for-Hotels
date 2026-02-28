# As a design-choice to increase the user-experience when importing the package, all the classes
# are summarized within one .py module. 
# The  following code can be used though in the case that the package should get more modules in the future:

# import importlib
# from pathlib import Path

# # 1. Defining the directory where this file (__init__.py) is located in
# package_dir = Path(__file__).parent

# # 2. Defining all => Those modules will be imported if "from package_name import *" is used.
# __all__ = []

# # 3. Importing all the modules within that directory (= package):

# for file_path in package_dir.glob("*.py"):
    
#     # Don't import the __init__ itself.
#     if file_path.name == "__init__.py":
#         continue
    
#     # Extracting the name of the module
#     module_name = file_path.stem

#     # Importing all the modules in the package
#     importlib.import_module(f".{module_name}", __package__)

#     # Adding module_name to __all__
#     __all__.append(module_name)


# ----------------------
# Explaination
# ----------------------
# package_dir.glob("*.py")
# Searches for the pattern "*.py" where * is the wildcard which can be any character of any length. 
# Returns the a Path for each .py-module.

# file_path.stem
# Extracts the modules name without the .py extension.

# __package__
# returns only the package name (= name of folder where the current file is saved).
# __package__ is only defined if __name__ != __main__ => Relative imports within a __main__ file is not
# possible.

# Relative Import: The "." only says: "Import this module (module after the dot) and search for it in the same
# folder than I am." And that folder must be defined with __package__.

#! Important: The combination of relative import + package doesn't return a path, but a theoretical structure:
#! Example: importlib.import_module(f".{module_name}", __package__) returns __package__.module_name which is 
#! interpreted as a structure of a path .../package/module_name and that path structure is searched within
#! sys.path paths (= Namespace).
#! sys.path are all the paths where Python searches for modules -> including the CWD!

# import_module(f".{module_name}", __package__)
# Imports the module_name from the package __package__