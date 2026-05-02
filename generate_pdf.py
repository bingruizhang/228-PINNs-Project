from markdown_pdf import Section, MarkdownPdf

pdf = MarkdownPdf(toc_level=0)
pdf.add_section(Section(open("docs/proposal_draft.md").read()))
pdf.save("docs/proposal.pdf")
