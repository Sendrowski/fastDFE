"""
Merge the lines of several files into one file.
"""

if @isdefined snakemake
    input = snakemake.input
    out = snakemake.output[1]
else
    input = Dict(i => "results/test/remote/est-sfs/passed.biallelic.90.probs." * string(i) * ".4.txt" for i in 1:4)
    out = "scratch/probs.txt"
end

# iterate over all files and write one line per file for each iteration
function gather_lines(f_chunks)
    open(out, "w") do f_out
        while true
            # iterate over file handles and write lines
            for f in f_chunks
                line = readline(f)

                if length(line) > 0
                    write(f_out, line * "\n")
                else
                    return
                end
            end
        end
    end
end

# initialize file handles
# make sure they are ordered correctly
f_chunks = [open(f) for f in [input[i] for i in 1:length(input)]]

gather_lines(f_chunks)

# close file handles
[close(f) for f in f_chunks]