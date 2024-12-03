# IR Interpol Pipeline
A proof of concept project of a Machine Learning Pipeline for generating room impulse responses. Also the code that ran the experiments for my Master Thesis.

A Dataset for testing the code can be downloaded from my Google Drive. Just unzip it and place the /data and /pickle dirs into the /ir_interpol_pipeline dir. Check the start-scripts for ideas on how to run the different experiments. The last line in the scripts runs the actual python program with arguments. Also you need a Python Env with:
<ol>
    <li>pandas==1.5 (or any 1.x -> pandas >=2 won't work)</li>
    <li>numpy==1.26.4 (must be compatible with pandas)</li>
    <li>librosa</li>
    <li>scipy==1.13 (>=1.14 breaks librosa, see https://github.com/librosa/librosa/issues/1849)</li>
    <li>mat73</li>
    <li>matplotlib</li>
    <li>tensorflow</li>
</ol>

## Link to Dataset:
https://drive.google.com/file/d/1H_mAVHuX44GsETmyoNGk9TLfl3n3MoHK/view?usp=sharing 
