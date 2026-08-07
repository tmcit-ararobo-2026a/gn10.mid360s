# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Extension template for scripting workflow."""

import gc

import omni
from isaacsim.gui.components.menu import MenuItemDescription
from omni.kit.menu.utils import add_menu_items, remove_menu_items

from .global_variables import EXTENSION_TITLE


class Extension(omni.ext.IExt):
    """Extension class for the scripting workflow template."""

    def on_startup(self, ext_id: str) -> None:
        """Initialize extension and UI elements.

        Args:
            ext_id: Extension identifier provided by the extension manager.
        """
        self.ext_id = ext_id

        action_registry = omni.kit.actions.core.get_action_registry()
        action_registry.register_action(
            ext_id,
            f"Create Mid-360S",
            self._menu_callback,
        )
        self._menu_items = [
            MenuItemDescription(name=EXTENSION_TITLE, onclick_action=(ext_id, f"Create Mid-360S"))
        ]

        add_menu_items(self._menu_items, "Create/Sensors")

    def on_shutdown(self) -> None:
        """Clean up resources on extension shutdown."""
        remove_menu_items(self._menu_items, "Create/Sensors")

        action_registry = omni.kit.actions.core.get_action_registry()
        action_registry.deregister_action(self.ext_id, f"Create Mid-360S")

        gc.collect()
        
    def _menu_callback(self) -> None:
        """Callback function for menu item click event."""
        # Add Mid-360S
        print(f"{EXTENSION_TITLE} Adding Mid-360S.")
