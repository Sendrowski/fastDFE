"""
Precompute polyDFE inference with divergence for the example datasets and cache the serialized
results (including the McDonald-Kreitman style ``alpha_divergence``) so fastDFE can be validated
against them offline. Mirrors ``infer_dfe_polydfe.py`` but enables divergence and stores the MK
alpha. Run from the repo root (paths are relative to it).
"""
import os
import warnings; warnings.filterwarnings('ignore')

import fastdfe as fd
from fastdfe.polydfe import PolyDFE
from fastdfe.spectrum import parse_polydfe_sfs_config
from testing.test_polydfe_wrapper import find_runnable_polydfe_binary

PP = 'resources/polydfe/postprocessing/script.R'
BIN = find_runnable_polydfe_binary()
assert BIN, 'no runnable polyDFE binary'

for ex in ['example_1', 'example_2', 'example_3']:
    sp = parse_polydfe_sfs_config(f'resources/polydfe/{ex}/spectra/sfs.txt')

    # run both modes from the *same* settings so the divergence-on vs divergence-off comparison is
    # matched (only the use of divergence differs)
    for include_divergence, suffix in [(True, 'divergence'), (False, 'nodivergence')]:
        config = fd.Config.from_file(f'testing/cache/configs/{ex}_C_full_anc/config.yaml')
        config.data['fixed_params']['all']['h'] = 0.5
        config.data['do_bootstrap'] = False
        config.data['include_divergence'] = include_divergence
        config.data['sfs_neut'] = fd.Spectra.from_spectrum(sp['sfs_neut'])
        config.data['sfs_sel'] = fd.Spectra.from_spectrum(sp['sfs_sel'])

        out_dir = f'testing/cache/polydfe/{ex}_C_full_anc_{suffix}'
        os.makedirs(out_dir, exist_ok=True)
        out_txt = f'{out_dir}/out.txt'

        polydfe = PolyDFE.from_config(config)
        polydfe.run(output_file=out_txt, binary=BIN, postprocessing_source=PP)
        if include_divergence:
            polydfe.get_alpha_divergence()  # computes and caches the MK alpha into the summary
        polydfe.to_file(f'{out_dir}/serialized.json')

        # the raw polyDFE output is not needed for the offline comparison
        if os.path.exists(out_txt):
            os.remove(out_txt)

        alpha = polydfe.summary.data.get('alpha_divergence', polydfe.summary.data['alpha'])
        print(f'CACHED {ex} ({suffix}): alpha={alpha:.4f}', flush=True)
