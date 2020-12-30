#!/usr/bin/env python

"""
ZetCode wxPython tutorial

This program draws a line on the
frame window after a while.

author: Jan Bodnar
website: zetcode.com
last edited: May 2018
"""

import wx
from wx.core import DC, Bitmap


class Example(wx.Frame):

    def __init__(self, *args, **kw):
        super(Example, self).__init__(*args, **kw)

        self.InitUI()

    def InitUI(self):

        wx.CallLater(0, self.DrawLine)

        self.SetTitle("Line")
        self.Centre()

    def DrawLine(self):

        #dc = wx.ClientDC(self)
        dc: DC = wx.ScreenDC()

        # Make screenshot

        app = wx.App(False)
        screen = wx.ScreenDC()

        size = screen.GetSize()
        width = size.width
        height = size.height
        bmp = wx.Bitmap(width, height)

        # Create a memory DC that will be used for actually taking the screenshot
        memDC = wx.MemoryDC()
        # Tell the memory DC to use our Bitmap
        # all drawing action on the memory DC will go to the Bitmap now
        memDC.SelectObject(bmp)
        # Blit (in this case copy) the actual screen on the memory DC
        memDC.Blit(
            0, 0,
            width, height,
            screen,
            0, 0
        )
        # Select the Bitmap out of the memory DC by selecting a new bitmap
        memDC.SelectObject(wx.NullBitmap)
        im = bmp.ConvertToImage()
        im.SaveFile('screenshot.png', wx.BITMAP_TYPE_PNG)

        # Draw something
        dc.DrawLine(50, 60, 190, 60)
        dc.DrawPolygon(((100, 100), (100, 300), (300, 300), (300, 100)))


def main():

    app = wx.App(False)
    frame = Example(None, wx.ID_ANY, "")
    app.MainLoop()


if __name__ == '__main__':
    main()