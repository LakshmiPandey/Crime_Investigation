# Generated by Django 2.0.2 on 2018-05-06 13:44

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('UI', '0002_auto_20180506_1913'),
    ]

    operations = [
        migrations.RenameField(
            model_name='facedatabase',
            old_name='dod',
            new_name='crimes',
        ),
    ]