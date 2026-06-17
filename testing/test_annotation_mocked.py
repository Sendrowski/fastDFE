"""
Drive the per-variant annotation pipeline (``DegeneracyAnnotation.annotate_site`` and its codon
parsing) directly with synthetic, integer-free inputs -- a hand-built reference contig, a CDS
record and ``DummyVariant`` sites -- instead of a real VCF/FASTA/GFF. This reaches the genome-
dependent annotation code that the end-to-end tests (inference/slow tier) need the full betula
genome for, while staying instantaneous.

Reference contig (1-based), single CDS on the + strand, phase 0:

    pos:  1 2 3 | 4 5 6 | 7 8 9 | 10 11 12 ...
    base: A T G | G T T | G T A |  G  T  C ...
    aa:    Met  |  Val  |  Val  |    Val
"""
import pandas as pd
import pytest

from fastdfe.annotation import DegeneracyAnnotation, SynonymyAnnotation
from fastdfe.io_handlers import DummyVariant

# Met ATG, then Val GTT/GTA/GTC..., Arg CGG, Pro CCC, Lys AAA, stop TAA
CONTIG = 'ATGGTTGTAGTCGTGGTACGGCCCAAATAA'


class _Handler:
    @staticmethod
    def get_aliases(chrom):
        return {chrom}


def _make_ann(strand='+', phase=0, start=1, end=len(CONTIG)):
    ann = DegeneracyAnnotation()
    ann._handler = _Handler()
    ann._cd = pd.Series({'seqid': 'chr1', 'start': start, 'end': end, 'strand': strand, 'phase': phase})
    ann._cd_prev = None
    ann._cd_next = None
    ann._contig = CONTIG
    ann._fetch = lambda v: None  # state is pre-injected, skip the file-based CDS fetch
    return ann


def _annotate(ann, ref, pos):
    v = DummyVariant(ref=ref, pos=pos, chrom='chr1')
    ann.annotate_site(v)
    return v


def test_degeneracy_fourfold_site():
    # pos 6 is the 3rd position of the Val codon GTT -> 4-fold degenerate
    ann = _make_ann()
    v = _annotate(ann, ref='T', pos=6)
    assert v.INFO['Degeneracy'] == 4
    assert v.INFO['Degeneracy_Info'] == '2,+,GTT'
    assert ann.n_annotated == 1


def test_degeneracy_zerofold_site():
    # pos 4 is the 1st position of the Val codon GTT -> 0-fold degenerate
    ann = _make_ann()
    v = _annotate(ann, ref='G', pos=4)
    assert v.INFO['Degeneracy'] == 0


def test_degeneracy_minus_strand_site():
    # on the minus strand the codon is read complemented from the CDS end; at pos 28 the codon is
    # 'TTA' (Leu) and the variant sits at its 3rd position -> 2-fold degenerate
    ann = _make_ann(strand='-')
    v = _annotate(ann, ref='T', pos=28)
    assert v.INFO['Degeneracy'] == 2
    assert v.INFO['Degeneracy_Info'] == '2,-,TTA'


def test_degeneracy_reference_mismatch_recorded():
    # ref allele 'A' does not match the contig base 'T' at pos 6 -> recorded as a mismatch, not annotated
    ann = _make_ann()
    v = _annotate(ann, ref='A', pos=6)
    assert v in ann.mismatches
    assert v.INFO['Degeneracy'] == '.'
    assert ann.n_annotated == 0


def test_degeneracy_skips_indel():
    # multi-base REF (indel) is skipped
    ann = _make_ann()
    _annotate(ann, ref='AT', pos=6)
    assert ann.n_skipped == 1


def test_degeneracy_skips_on_missing_cds():
    # a LookupError from the CDS fetch (no overlapping CDS) skips the site
    ann = _make_ann()
    ann._fetch = lambda v: (_ for _ in ()).throw(LookupError('no cds'))
    v = _annotate(ann, ref='T', pos=6)
    assert ann.n_skipped == 1
    assert v.INFO['Degeneracy'] == '.'


# --------------------------------------------------------------------------- synonymy annotation

def _make_syn(strand='+'):
    ann = SynonymyAnnotation()
    ann._handler = _Handler()
    ann._cd = pd.Series({'seqid': 'chr1', 'start': 1, 'end': len(CONTIG), 'strand': strand, 'phase': 0})
    ann._cd_prev = None
    ann._cd_next = None
    ann._contig = CONTIG
    ann._fetch = lambda v: None
    return ann


def _snp(ref, alt, pos):
    v = DummyVariant(ref=ref, pos=pos, chrom='chr1')
    v.is_snp = True
    v.ALT = [alt]
    return v


def test_synonymy_synonymous_change():
    # GTT -> GTC keeps the amino acid Valine -> synonymous (Synonymy == 1)
    ann = _make_syn()
    v = _snp('T', 'C', 6)
    ann.annotate_site(v)
    assert v.INFO['Synonymy'] == 1
    assert v.INFO['Synonymy_Info'] == 'GTT/GTC'


def test_synonymy_nonsynonymous_change():
    # GTT -> ATT changes Valine to Isoleucine -> non-synonymous (Synonymy == 0)
    ann = _make_syn()
    v = _snp('G', 'A', 4)
    ann.annotate_site(v)
    assert v.INFO['Synonymy'] == 0
    assert v.INFO['Synonymy_Info'] == 'GTT/ATT'


def test_synonymy_skips_non_snp():
    # a non-SNP (mono-allelic) site is not annotated for synonymy
    ann = _make_syn()
    v = DummyVariant(ref='T', pos=6, chrom='chr1')  # DummyVariant.is_snp is False
    ann.annotate_site(v)
    assert v.INFO['Synonymy'] == '.'
