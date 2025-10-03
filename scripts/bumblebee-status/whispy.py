"""Display whispy recording status"""

import core.module
import core.widget
import core.decorators
import subprocess

class Module(core.module.Module):
    @core.decorators.every(seconds=1)
    def __init__(self, config, theme):
        super().__init__(config, theme, core.widget.Widget(self.status))

    def status(self, widget):
        try:
            # Check if pw-record is running (means recording is active)
            result = subprocess.run(['pgrep', '-x', 'pw-record'],
                                  capture_output=True,
                                  timeout=1)
            if result.returncode == 0:
                return "🔴REC"
            else:
                return "⚪"
        except:
            return "?"
