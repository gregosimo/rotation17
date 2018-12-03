figdir = fig
latexfigures := $(wildcard $(figdir))

Bruntt_comp.pdf: paperexport.py
	python paperexport.py Bruntt

Pleiades_comp.pdf: paperexport.py
	python paperexport.py SH

astero.pdf: paperexport.py
	python paperexport.py asterosamp

cool_samp.pdf: paperexport.py
	python paperexport.py coolsamp

astero_rot.pdf: paperexport.py
	python paperexport.py asterorot

detection_fraction.pdf: paperexport.py
	python paperexport.py rrfracs

cool_rot.pdf: paperexport.py
	python paperexport.py coolrot

figures: paperexport.py
	python paperexport.py all

tabledir = tables
tablelist := $(wildcard $(tabledir)/*)
latextables := $(wildcard $(tabledir/*.tex))

mainfile = main
maintex = $(mainfile).tex

bibfile = references.bib

revision = main_apj1.tex

all: $(mainfile).pdf

mnras: mnras.tar.gz
	
mnras.tar.gz: $(mainfile).pdf $(latexfigures) $(tablelist)
	tar -zcf mnras.tar.gz $(maintex) main.bbl $(bibfile) readme.mnras \
	    $(latexfigures) $(tablelist)

apj: apj.tar.gz


# Is there an automated way to add dependencies for tables and figures in here? 
# Ideally it should be read from the LaTeX file. Potentially stripped out, but
# that may be annoyingly difficult.
#
$(mainfile).pdf: $(maintex) $(bibfile) paperexport.py $(latexfigures) 
	latexmk -pdf $(maintex)

$(mainfile).ps: $(maintex) $(bibfile $(latexfigures))
	latexmk -ps $(maintex)

$(mainfile).bbl: $(maintex) $(bibfile)
	bibtex $(mainfile)

referee: diff.pdf
	
diff.pdf: $(revision) $(maintex)
	-rm diff.*
	latexdiff $(revision) $(maintex) > diff.tex
	latexmk -pdf -interaction=nonstopmode diff.tex
	mv diff.tex diff.pdf referee_material

diff.ps: $(revision) $(maintex)
	-rm diff.*
	latexdiff $(revision) $(maintex) > diff.tex
	latexmk -ps -interaction=nonstopmode diff.tex
	mv diff.tex diff.pdf referee_material

arxiv: arxiv.tar.gz

arxiv.tar.gz: $(maintex) $(mainfile).bbl $(latexfigures) $(latextables)
	tar -czf arxiv.tar.gz $(maintex) $(mainfile).bbl $(latexfigures) \
		$(latextables) 

.PHONY : clean
clean:
	-rm diff.*
	latexmk -C $(maintex)
