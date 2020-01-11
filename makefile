figdir = fig
latexfigures := $(wildcard $(figdir)/*)

tabledir = tables
tablelist := $(wildcard $(tabledir)/*)

mainfile = main
maintex = $(mainfile).tex

revision = main_080219.tex

bibfile = references.bib

all: $(mainfile).pdf

mnras: mnras.tar.gz
	
mnras.tar.gz: $(mainfile).pdf $(latexfigures) $(tablelist)
	tar -zcf mnras.tar.gz $(maintex) main.bbl references.bib readme.mnras \
	    $(latexfigures) $(tablelist)

aas: $(maintex) $(mainfile).pdf references.bib $(latexfigures) $(tablelist) 
	mkdir aas
	cp $(maintex) aas/$(maintex)
	cp $(maintex).pdf aas/
	cp $(latexfigures) aas/
	cp $(tablelist) aas/
	cp references.bib aas/


# Is there an automated way to add dependencies for tables and figures in here? 
# Ideally it should be read from the LaTeX file. Potentially stripped out, but
# that may be annoyingly difficult.
#
$(mainfile).pdf: $(maintex) $(bibfile) $(tablelist) $(latexfigures)
	latexmk -pdf $(maintex)

$(mainfile).ps: $(maintex) $(bibfile)
	latexmk -ps $(maintex)

$(mainfile).bbl: $(maintex) $(bibfile)
	bibtex $(mainfile)

referee: diff.pdf
	
diff.pdf: $(revision) $(maintex) $(bibfile)
	-rm diff.*
	latexdiff $(revision) $(maintex) > diff.tex
	latexmk -pdf -interaction=nonstopmode diff.tex
#mv diff.tex diff.pdf referee_material

diff.ps: $(revision) $(maintex) $(bibfile)
	-rm diff.*
	latexdiff $(revision) $(maintex) > diff.tex
	latexmk -ps -interaction=nonstopmode diff.tex
	mv diff.tex diff.pdf referee_material

arxiv: arxiv.tar.gz

arxiv.tar.gz: $(maintex) $(mainfile).bbl $(latexfigures) $(tablelist)
	tar -czf arxiv.tar.gz $(maintex) $(mainfile).bbl $(latexfigures) \
		$(tablelist) mnras.cls

.PHONY : clean
clean:
	-rm diff.*
	latexmk -C $(maintex)
	-rm -r aas
