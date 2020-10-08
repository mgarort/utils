import os
import matplotlib.pyplot as plt
import glob


def load_style(stylename):
    '''
    This function loads matplotlib styles, which could be either default or customized.
    '''

    def get_stylename_from_stylepath(stylepath):
        stylename = stylepath.split('.mplstyle')[0]
        stylename = os.path.split(stylename)[1]
        return(stylename)

    # If stylename is a custom style
    __import__('pdb').set_trace()
    stylelib_path = os.path.join(os.path.dirname(__file__), 'stylelib')
    custom_stylepaths = glob.glob(os.path.join(stylelib_path,'*.mplstyle'))
    custom_stylenames = [get_stylename_from_stylepath(stylepath) for stylepath in custom_stylepaths]
    custom_styles = {key: value for key, value in zip(custom_stylenames, custom_stylepaths)}
    if stylename in custom_stylenames:
        this_style_path = custom_styles[stylename]
        plt.style.use(this_style_path)

    # If stylename is a default style
    elif stylename in plt.style.available:
        plt.style.use(stylename)

    else:
        raise ValueError(stylename, 'is not an available style.')



