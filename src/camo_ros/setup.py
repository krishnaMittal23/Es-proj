from setuptools import setup
import os
from glob import glob
package_name = 'camo_ros'
setup(
    name=package_name, version='0.1.0', packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'], zip_safe=True,
    entry_points={'console_scripts': [
        'detector_node = camo_ros.detector_node:main',
        'mission_node = camo_ros.mission_node:main',
        'viewer_node = camo_ros.viewer_node:main',
        'dataset_node = camo_ros.dataset_node:main',
    ]},
)
