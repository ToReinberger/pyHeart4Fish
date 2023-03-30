from setuptools import setup

setup(
    name='pyHeart4Fish',
    version='0.0.1',
    packages=['pyHeart4Fish_python',
              'pyHeart4Fish_python.Extra_scripts',
              'pyHeart4Fish_python.Logo',
              'pyHeart4Fish_python.Test_data_github',
              'pyHeart4Fish_python.Test_data_github.avi_files_Results_example',
              'pyHeart4Fish_python.Test_data_github.avi_files_Results_example.fish_1',
              'pyHeart4Fish_python.Test_data_github.avi_files_Results_example.fish_2',
              'pyHeart4Fish_python.Test_data_github.czi-files_Results_example',
              'pyHeart4Fish_python.Test_data_github.czi-files_Results_example.fish_1',
              'pyHeart4Fish_python.Test_data_github.czi-files_Results_example.fish_2',
              'pyHeart4Fish_python.Test_data_github.tif_files_Results_example',
              'pyHeart4Fish_python.Test_data_github.tif_files_Results_example.fish_1',
              'pyHeart4Fish_python.Test_data_github.tif_files_Results_example.fish_2',
              ],
    url='https://github.com/ToReinberger/pyHeart4Fish_GUI',
    license='BSD 2-Clause',
    author='tobiasreinberger',
    author_email='tobias.reinberger@uni-luebeck.de',
    description='pyHeart4Fish_Heartbeat_analysis_tool',
    python_requires=">=3.7, <4",
    install_requires=["aicsimageio",
                      "czifile",
                      "aicspylibczi",
                      "matplotlib",
                      "numpy",
                      "opencv_python",
                      "pandas",
                      "Pillow",
                      "scipy",
                      "openpyxl",
                      "pyshortcuts"
                      ]
)


