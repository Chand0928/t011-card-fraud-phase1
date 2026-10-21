"""Execute each notebook code cell with IPython and preserve actual rich outputs.

This local runner needs no notebook server, open sockets or kernel registration.
The resulting notebook also runs normally in Jupyter/VS Code.
"""
from pathlib import Path
import os
import sys
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
path = ROOT / 'notebooks/phase1_gmm.ipynb'
nb = nbformat.read(path, as_version=4)
shell = InteractiveShell.instance()
count = 0
for cell in nb.cells:
    if cell.cell_type != 'code':
        continue
    count += 1
    with capture_output() as captured:
        result = shell.run_cell(cell.source, store_history=True)
    if result.error_before_exec or result.error_in_exec:
        raise RuntimeError(f'Notebook cell {count} failed: {captured.stdout} {captured.stderr}')
    outputs = []
    if captured.stdout:
        outputs.append(nbformat.v4.new_output('stream', name='stdout', text=captured.stdout))
    if captured.stderr:
        outputs.append(nbformat.v4.new_output('stream', name='stderr', text=captured.stderr))
    for item in captured.outputs:
        outputs.append(nbformat.v4.new_output('display_data', data=item.data, metadata=item.metadata))
    cell.execution_count = count
    cell.outputs = outputs
nb.metadata['execution'] = {'method': 'IPython run_cell, sequential, captured actual rich outputs',
                            'all_code_cells_succeeded': True}
nbformat.validate(nb)
nbformat.write(nb, path)
print(f'Executed {count} code cells successfully and saved outputs.')
