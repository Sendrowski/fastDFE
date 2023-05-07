"""
Scatter the lines of a file into several files by writing
the jth line into the `jth % n_files` files.
"""

if @isdefined snakemake
    input = snakemake.input[1]
    n_chunks = snakemake.params["n_chunks"]
    i = snakemake.params["i"]
    out = snakemake.output[1]
else
    input = "results/test/remote/est-sfs/passed.biallelic.90.data.all.txt"
    n_chunks = 4
    i = 3
    out = "scratch/$i.data.txt"
end

open(input) do f_in
    open(out, "w") do f_out
        for (j, line) in enumerate(eachline(f_in))
            # only write every 'j % n == i % n'th line
            if j % n_chunks == i % n_chunks
                write(f_out, line * "\n")
            end
        end
    end
end