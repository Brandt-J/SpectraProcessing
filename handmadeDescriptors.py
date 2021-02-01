"""
SPECTRA PROCESSING
Copyright (C) 2020 Josef Brandt, University of Gothenborg.
<josef.brandt@gu.se>

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program, see COPYING.
If not, see <https://www.gnu.org/licenses/>.
"""


from descriptors import DescriptorSet, DescriptorLibrary
__all__ = ['handMadeDescLib']

handMadeDescLib: DescriptorLibrary = DescriptorLibrary()


desc_HDPE: DescriptorSet = DescriptorSet('HDPE')
desc_HDPE.add_descriptor(700, 720, 740)
desc_HDPE.add_descriptor(1435, 1470, 1485)
desc_HDPE.add_descriptor(2825, 2852, 2870)
desc_HDPE.add_descriptor(2870, 2925, 2965)
handMadeDescLib.add_descriptorSet(desc_HDPE)


# desc_LDPE: DescriptorSet = DescriptorSet('LDPE')
# desc_LDPE.add_descriptor(700, 720, 740)
# desc_LDPE.add_descriptor(1361, 1377, 1390)  # The characteristic LDPE band!
# desc_LDPE.add_descriptor(1435, 1470, 1485)
# desc_LDPE.add_descriptor(2825, 2852, 2870)
# desc_LDPE.add_descriptor(2870, 2925, 2965)
# handMadeDescLib.add_descriptorSet(desc_LDPE)


desc_PA6: DescriptorSet = DescriptorSet('PA6')  # according to Ksouri et al, J Polym Res 2017, 24: 133
desc_PA6.add_descriptor(1220, 1260, 1300)  # Amide III
desc_PA6.add_descriptor(1500, 1550, 1600)  # Amide II
desc_PA6.add_descriptor(1600, 1650, 1700)  # Amide I
desc_PA6.add_descriptor(2825, 2852, 2870)
# desc_PA6.add_descriptor(2870, 2925, 2965)
desc_PA6.add_descriptor(3020, 3080, 3130)  # N-H stretching
desc_PA6.add_descriptor(3200, 3300, 3400)  # Amide A (hydrogen bonded N-H)
handMadeDescLib.add_descriptorSet(desc_PA6)


desc_PC: DescriptorSet = DescriptorSet('PC')
desc_PC.add_descriptor(980, 1015, 1040)  # O-C-O stretch
desc_PC.add_descriptor(1060, 1080, 1095)  # C-C-C deformation
desc_PC.add_descriptor(1250, 1260, 1275)  # aromatic ether stretch
desc_PC.add_descriptor(1475, 1505, 1550)  # C-C stretch Ring
desc_PC.add_descriptor(1690, 1770, 1840)  # C=O stretch
desc_PC.add_descriptor(2950, 2970, 3020)
desc_PC.add_descriptor(3020, 3040, 3050)
handMadeDescLib.add_descriptorSet(desc_PC)


desc_PET: DescriptorSet = DescriptorSet('PET')  # Assignments according to Andanson, Kazarian, Macromol. Symp.
desc_PET.add_descriptor(700, 725, 750)  # oop benzene group
desc_PET.add_descriptor(820, 845, 860)  # CH2 rocking of glycol  # Maybe add in again? But it's a quite small band
desc_PET.add_descriptor(860, 872, 890)  # oop benzene group
desc_PET.add_descriptor(935, 970, 1000)  # C-O stretching glycol
desc_PET.add_descriptor(1000, 1020, 1050)  # in-plane benzene group
desc_PET.add_descriptor(1320, 1340, 1360)  # CH2 wagging of glycol
desc_PET.add_descriptor(1716, 1741, 1760)
desc_PET.add_descriptor(2870, 2910, 2930)
desc_PET.add_descriptor(2930, 2970, 3030)
desc_PET.add_descriptor(3035, 3070, 3090)
handMadeDescLib.add_descriptorSet(desc_PET)


# BETTER SPEC IS NEEDED! ASSIGNMENTS CAN BE FOUND IN:
# Namouchi et al, "Investigation on electrical properties of thermally aged PMMA by combined
# use of FTIR and impedance spectroscopies" 10.1016/j.jallcom.2008.01.148
# desc_PMMA: DescriptorSet = DescriptorSet('PMMA')  #
# desc_PMMA.add_descriptor(1675, 1725, 1770)


desc_PP: DescriptorSet = DescriptorSet('PP')
desc_PP.add_descriptor(795, 807, 820)
desc_PP.add_descriptor(830, 840, 849)
desc_PP.add_descriptor(962, 973, 980)
desc_PP.add_descriptor(988, 996, 1005)
desc_PP.add_descriptor(1134, 1166, 1180)
desc_PP.add_descriptor(1337, 1375, 1387)
desc_PP.add_descriptor(1422, 1452, 1480)
desc_PP.add_descriptor(2860, 2870, 2885)
desc_PP.add_descriptor(2890, 2920, 2940)
desc_PP.add_descriptor(2940, 2950, 2980)
handMadeDescLib.add_descriptorSet(desc_PP)


desc_PS: DescriptorSet = DescriptorSet('PS')
desc_PS.add_descriptor(1590, 1600, 1610)
desc_PS.add_descriptor(2825, 2850, 2870)
desc_PS.add_descriptor(3010, 3025, 3043)
desc_PS.add_descriptor(3043, 3061, 3071)
desc_PS.add_descriptor(3071, 3082, 3096)
handMadeDescLib.add_descriptorSet(desc_PS)
