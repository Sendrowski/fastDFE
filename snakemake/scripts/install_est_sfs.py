"""
Set up est-sfs by downloading it from Sourceforge and compiling it.
"""

__author__ = "Janek Sendrowski"
__contact__ = "j.sendrowski18@gmail.com"
__date__ = "2022-05-31"

from snakemake.shell import shell

try:
    out = snakemake.output[0]
    tmp_dir = snakemake.resources.tmpdir
    max_sites = snakemake.config['max_sites']
except NameError:
    # testing
    out = "scratch/est-sfs"
    tmp_dir = "/tmp"
    max_sites = 100000

version = "2.04"

# set up est-sfs
# Note: the compilation only works on Linux
shell(f"""
    set -x
    work_dir=$(readlink -f .)
    cd {tmp_dir}
    
    wget https://sourceforge.net/projects/est-usfs/files/est-sfs-release-{version}.tar.gz/download -O est-sfs.tar.gz
    
    rm -rf est-sfs
    mkdir est-sfs
    tar -xvf est-sfs.tar.gz -C est-sfs --strip-components 1
    
    cd est-sfs
    sed -i 's/#define max_config 1000000/#define max_config {max_sites}/g' est-sfs.c
    export C_INCLUDE_PATH=$CONDA_PREFIX/include
    make
    
    mv est-sfs "$work_dir/{out}"
""")
