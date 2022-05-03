import jinja2

autoescape = jinja2.select_autoescape(default=True, default_for_string=True)


# For _repr_html_()
repr_templates_env = jinja2.Environment(
    loader=jinja2.PackageLoader(
        package_name="pycrostates.html_templates", package_path="repr"
    ),
    autoescape=autoescape,
)
