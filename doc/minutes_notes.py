"""
Deomo how to creating PDF Files with Python

url info: https://towardsdatascience.com/creating-pdf-files-with-python-ad3ccadfae0f

"""

from fpdf import FPDF

# size of A4 (w:210 mm and h:297 mm)
pdf_w = 210
pdf_h = 297


class PDF(FPDF):
    def line(self):
        """ add outline on the page"""
        self.set_line_width(0.0)
        self.line(5.0, 5.0, 205.0, 5.0)  # top one
        self.line(5.0, 292.0, 205.0, 292.0)  # bottom one
        self.line(5.0, 5.0, 5.0, 292.0)  # left one
        self.line(205.0, 5.0, 205.0, 292.0)  # right one

    def lines(self):
        """ add two outlines on the page"""
        self.rect(5.0, 5.0, 200.0, 287.0)
        self.rect(8.0, 8.0, 194.0, 282.0)

    def image_logo(self):
        """add image (stockholm log) to the top right side"""
        self.set_xy(183.0, 10.0) # set position on page
        self.image('../images/stockholm-university-logo_mini.png',
                   link='', type='', w=1586 / 80, h=1586 / 80)

    def titles(self, title):
        """add title in the center with colour dark blue"""
        self.set_xy(0.0, 0.0)
        self.set_font('Arial', 'B', 16)
        self.set_text_color(0, 51, 102)
        self.cell(w=210.0, h=40.0, align='C', txt=title, border=0)

    def charts(self):
        self.set_xy(40.0, 25.0)
        self.image(os.getcwd() + '/' + "pltx.png", link='', type='', w=700 / 5, h=450 / 5)


if __name__ == "__main__":
    '''
    create an object of the PDF class
    Parameters of FPDF constructor:
    orientation : page orientation = default: “portrait” P
                                    other : “landscape” L 
    unit: the unit of measurement = default:  “millimeter” mm
                                    other : “centimeter,” “points,” or “inches.” 
    format: page format = default : “A4”
                          Other : “A3”, “A5”, “Letter,” and “Legal”
    '''

    pdf = PDF()

    pdf.add_page()  # add a new page to the document

    pdf.image_logo()

    pdf.titles(title='Minutes Notes')

    pdf.output('minutes_notes.pdf', 'F')  # save the output.
